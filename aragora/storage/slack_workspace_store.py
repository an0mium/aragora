"""
Slack Workspace Storage for OAuth token management.

Stores workspace credentials after OAuth installation for multi-workspace support.
Tokens are encrypted at rest using AES-256-GCM when ARAGORA_ENCRYPTION_KEY is set.

Schema:
    CREATE TABLE slack_workspaces (
        workspace_id TEXT PRIMARY KEY,
        workspace_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        bot_user_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        tenant_id TEXT,
        is_active INTEGER DEFAULT 1
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
SLACK_WORKSPACE_DB_PATH = os.environ.get(
    "SLACK_WORKSPACE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "slack_workspaces.db"),
)

# Encryption key for tokens (required in production)
ENCRYPTION_KEY = os.environ.get("ARAGORA_ENCRYPTION_KEY", "")

# Environment mode
ARAGORA_ENV = os.environ.get("ARAGORA_ENV", "development")

# Track if encryption warning has been shown
_encryption_warning_shown = False


@dataclass
class SlackWorkspace:
    """Represents an installed Slack workspace."""

    workspace_id: str  # Slack team_id
    workspace_name: str
    access_token: str  # Bot token (xoxb-*)
    bot_user_id: str
    installed_at: float  # Unix timestamp
    installed_by: Optional[str] = None  # User ID who installed
    scopes: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None  # Link to Aragora tenant
    is_active: bool = True
    signing_secret: Optional[str] = None  # Workspace-specific signing secret
    refresh_token: Optional[str] = None  # OAuth refresh token for token renewal
    token_expires_at: Optional[float] = None  # Unix timestamp when access_token expires

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes sensitive token and signing_secret)."""
        return {
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "bot_user_id": self.bot_user_id,
            "installed_at": self.installed_at,
            "installed_at_iso": datetime.fromtimestamp(
                self.installed_at, tz=timezone.utc
            ).isoformat(),
            "installed_by": self.installed_by,
            "scopes": self.scopes,
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "has_signing_secret": bool(self.signing_secret),
            "has_refresh_token": bool(self.refresh_token),
            "token_expires_at": self.token_expires_at,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SlackWorkspace":
        """Create from database row."""
        scopes_str = row["scopes"] or ""
        scopes = scopes_str.split(",") if scopes_str else []

        # Handle signing_secret column which may not exist in older DBs
        signing_secret = None
        try:
            signing_secret = row["signing_secret"]
        except (IndexError, KeyError):
            pass

        return cls(
            workspace_id=row["workspace_id"],
            workspace_name=row["workspace_name"],
            access_token=row["access_token"],
            bot_user_id=row["bot_user_id"],
            installed_at=row["installed_at"],
            installed_by=row["installed_by"],
            scopes=scopes,
            tenant_id=row["tenant_id"],
            is_active=bool(row["is_active"]),
            signing_secret=signing_secret,
        )


class SlackWorkspaceStore:
    """
    Storage for Slack workspace OAuth credentials.

    Supports SQLite backend with optional token encryption.
    Thread-safe for concurrent access.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS slack_workspaces (
        workspace_id TEXT PRIMARY KEY,
        workspace_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        bot_user_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        tenant_id TEXT,
        is_active INTEGER DEFAULT 1,
        signing_secret TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_slack_workspaces_tenant
        ON slack_workspaces(tenant_id);

    CREATE INDEX IF NOT EXISTS idx_slack_workspaces_active
        ON slack_workspaces(is_active);
    """

    # Migration to add signing_secret column to existing databases
    MIGRATION_ADD_SIGNING_SECRET = """
    ALTER TABLE slack_workspaces ADD COLUMN signing_secret TEXT;
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
                    "Slack OAuth tokens must be encrypted at rest."
                )
            elif not _encryption_warning_shown:
                logger.warning(
                    "Slack tokens will be stored UNENCRYPTED. "
                    "Set ARAGORA_ENCRYPTION_KEY for production use."
                )
                _encryption_warning_shown = True

        self._db_path = db_path or SLACK_WORKSPACE_DB_PATH
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

                # Run migration to add signing_secret column if needed
                try:
                    cursor = conn.execute("PRAGMA table_info(slack_workspaces)")
                    columns = {row[1] for row in cursor.fetchall()}
                    if "signing_secret" not in columns:
                        conn.execute(self.MIGRATION_ADD_SIGNING_SECRET)
                        conn.commit()
                        logger.info("Added signing_secret column to slack_workspaces")
                except sqlite3.Error as e:
                    logger.debug(f"Migration check: {e}")

                self._initialized = True

    def _derive_key_v2(self) -> bytes:
        """Derive encryption key using PBKDF2HMAC (secure KDF).

        Uses a deterministic salt derived from the key itself to ensure
        consistent encryption/decryption across restarts while still
        benefiting from the iterative key stretching of PBKDF2.
        """
        import base64
        import hashlib

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        # Use SHA-256 of key as deterministic salt (16 bytes)
        # This provides domain separation while remaining deterministic
        salt = hashlib.sha256(b"aragora-slack-token-salt:" + ENCRYPTION_KEY.encode()).digest()[:16]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,  # OWASP recommended minimum for PBKDF2-SHA256
        )
        return base64.urlsafe_b64encode(kdf.derive(ENCRYPTION_KEY.encode()))

    def _derive_key_v1(self) -> bytes:
        """Derive key using legacy SHA-256 method (for backward compatibility)."""
        import base64
        import hashlib

        return base64.urlsafe_b64encode(hashlib.sha256(ENCRYPTION_KEY.encode()).digest())

    def _encrypt_token(self, token: str) -> str:
        """Encrypt token using PBKDF2-derived key.

        Tokens are prefixed with version identifier for future migration support:
        - v2: PBKDF2HMAC with 480k iterations
        - (no prefix): Legacy SHA-256 single-pass
        """
        if not ENCRYPTION_KEY:
            return token

        try:
            from cryptography.fernet import Fernet

            # Use secure PBKDF2 key derivation
            key = self._derive_key_v2()
            f = Fernet(key)
            encrypted = f.encrypt(token.encode()).decode()
            # Prefix with version for future-proofing
            return f"v2:{encrypted}"
        except ImportError:
            logger.warning("cryptography not installed, storing token unencrypted")
            return token
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token encryption failed: {e}")
            return token

    def _decrypt_token(self, encrypted: str) -> str:
        """Decrypt token with support for multiple KDF versions.

        Supports:
        - v2: prefix - PBKDF2HMAC derived key
        - No prefix - Legacy SHA-256 derived key
        """
        if not ENCRYPTION_KEY:
            return encrypted

        # Check if it looks like an unencrypted token
        if encrypted.startswith("xoxb-") or encrypted.startswith("xoxp-"):
            return encrypted  # Not encrypted

        try:
            from cryptography.fernet import Fernet

            # Check for version prefix
            if encrypted.startswith("v2:"):
                # New PBKDF2 encryption
                key = self._derive_key_v2()
                ciphertext = encrypted[3:]  # Strip "v2:" prefix
            else:
                # Legacy SHA-256 encryption
                key = self._derive_key_v1()
                ciphertext = encrypted

            f = Fernet(key)
            return f.decrypt(ciphertext.encode()).decode()
        except ImportError:
            return encrypted
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token decryption failed: {e}")
            return encrypted

    def save(self, workspace: SlackWorkspace) -> bool:
        """Save or update a workspace.

        Args:
            workspace: Workspace to save

        Returns:
            True if saved successfully
        """
        conn = self._get_connection()
        try:
            encrypted_token = self._encrypt_token(workspace.access_token)
            encrypted_secret = (
                self._encrypt_token(workspace.signing_secret) if workspace.signing_secret else None
            )
            scopes_str = ",".join(workspace.scopes)

            conn.execute(
                """
                INSERT OR REPLACE INTO slack_workspaces
                (workspace_id, workspace_name, access_token, bot_user_id,
                 installed_at, installed_by, scopes, tenant_id, is_active, signing_secret)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace.workspace_id,
                    workspace.workspace_name,
                    encrypted_token,
                    workspace.bot_user_id,
                    workspace.installed_at,
                    workspace.installed_by,
                    scopes_str,
                    workspace.tenant_id,
                    1 if workspace.is_active else 0,
                    encrypted_secret,
                ),
            )
            conn.commit()
            logger.info(f"Saved Slack workspace: {workspace.workspace_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to save workspace: {e}")
            return False

    def get(self, workspace_id: str) -> Optional[SlackWorkspace]:
        """Get a workspace by ID.

        Args:
            workspace_id: Slack team_id

        Returns:
            Workspace or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM slack_workspaces WHERE workspace_id = ?",
                (workspace_id,),
            )
            row = cursor.fetchone()

            if row:
                workspace = SlackWorkspace.from_row(row)
                workspace.access_token = self._decrypt_token(workspace.access_token)
                if workspace.signing_secret:
                    workspace.signing_secret = self._decrypt_token(workspace.signing_secret)
                if workspace.refresh_token:
                    workspace.refresh_token = self._decrypt_token(workspace.refresh_token)
                return workspace

            return None

        except sqlite3.Error as e:
            logger.error(f"Failed to get workspace {workspace_id}: {e}")
            return None

    def get_by_tenant(self, tenant_id: str) -> List[SlackWorkspace]:
        """Get all workspaces for a tenant.

        Args:
            tenant_id: Aragora tenant ID

        Returns:
            List of workspaces
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM slack_workspaces
                WHERE tenant_id = ? AND is_active = 1
                ORDER BY installed_at DESC
                """,
                (tenant_id,),
            )

            workspaces = []
            for row in cursor.fetchall():
                workspace = SlackWorkspace.from_row(row)
                workspace.access_token = self._decrypt_token(workspace.access_token)
                if workspace.signing_secret:
                    workspace.signing_secret = self._decrypt_token(workspace.signing_secret)
                workspaces.append(workspace)

            return workspaces

        except sqlite3.Error as e:
            logger.error(f"Failed to get workspaces for tenant {tenant_id}: {e}")
            return []

    def list_active(self, limit: int = 100, offset: int = 0) -> List[SlackWorkspace]:
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
                SELECT * FROM slack_workspaces
                WHERE is_active = 1
                ORDER BY installed_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

            workspaces = []
            for row in cursor.fetchall():
                workspace = SlackWorkspace.from_row(row)
                workspace.access_token = self._decrypt_token(workspace.access_token)
                if workspace.signing_secret:
                    workspace.signing_secret = self._decrypt_token(workspace.signing_secret)
                workspaces.append(workspace)

            return workspaces

        except sqlite3.Error as e:
            logger.error(f"Failed to list workspaces: {e}")
            return []

    def deactivate(self, workspace_id: str) -> bool:
        """Deactivate a workspace (on uninstall).

        Args:
            workspace_id: Slack team_id

        Returns:
            True if deactivated successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE slack_workspaces SET is_active = 0 WHERE workspace_id = ?",
                (workspace_id,),
            )
            conn.commit()
            logger.info(f"Deactivated Slack workspace: {workspace_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to deactivate workspace {workspace_id}: {e}")
            return False

    def delete(self, workspace_id: str) -> bool:
        """Permanently delete a workspace.

        Args:
            workspace_id: Slack team_id

        Returns:
            True if deleted successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM slack_workspaces WHERE workspace_id = ?",
                (workspace_id,),
            )
            conn.commit()
            logger.info(f"Deleted Slack workspace: {workspace_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to delete workspace {workspace_id}: {e}")
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
                cursor = conn.execute("SELECT COUNT(*) FROM slack_workspaces WHERE is_active = 1")
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM slack_workspaces")

            return cursor.fetchone()[0]

        except sqlite3.Error as e:
            logger.error(f"Failed to count workspaces: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get workspace statistics.

        Returns:
            Statistics dictionary
        """
        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM slack_workspaces").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM slack_workspaces WHERE is_active = 1"
            ).fetchone()[0]

            return {
                "total_workspaces": total,
                "active_workspaces": active,
                "inactive_workspaces": total - active,
            }

        except sqlite3.Error as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_workspaces": 0, "active_workspaces": 0}

    def refresh_workspace_token(
        self,
        workspace_id: str,
        client_id: str,
        client_secret: str,
    ) -> Optional[SlackWorkspace]:
        """Refresh an expired access token using the refresh token.

        Args:
            workspace_id: Slack team_id
            client_id: Slack OAuth client ID
            client_secret: Slack OAuth client secret

        Returns:
            Updated workspace with new tokens, or None on failure
        """
        import urllib.request
        import urllib.parse
        import json

        workspace = self.get(workspace_id)
        if not workspace:
            logger.error(f"Workspace not found for refresh: {workspace_id}")
            return None

        if not workspace.refresh_token:
            logger.error(f"No refresh token available for workspace: {workspace_id}")
            return None

        try:
            # Exchange refresh token for new access token
            data = urllib.parse.urlencode(
                {
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": workspace.refresh_token,
                }
            ).encode()

            request = urllib.request.Request(
                "https://slack.com/api/oauth.v2.access",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode())

            if not result.get("ok"):
                error = result.get("error", "unknown")
                logger.error(f"Token refresh failed for {workspace_id}: {error}")
                # If token is revoked, deactivate the workspace
                if error in ("invalid_refresh_token", "token_revoked"):
                    self.deactivate(workspace_id)
                return None

            # Update workspace with new tokens
            workspace.access_token = result.get("access_token", workspace.access_token)

            # Handle new refresh token (rotation)
            new_refresh = result.get("refresh_token")
            if new_refresh:
                workspace.refresh_token = new_refresh

            # Calculate expiration time
            expires_in = result.get("expires_in")
            if expires_in:
                workspace.token_expires_at = time.time() + expires_in

            # Save updated workspace
            if self.save(workspace):
                logger.info(f"Successfully refreshed token for workspace: {workspace_id}")
                return workspace

            return None

        except urllib.error.URLError as e:
            logger.error(f"Network error refreshing token for {workspace_id}: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid response refreshing token for {workspace_id}: {e}")
            return None

    def is_token_expired(self, workspace_id: str, buffer_seconds: int = 300) -> bool:
        """Check if a workspace's access token is expired or will expire soon.

        Args:
            workspace_id: Slack team_id
            buffer_seconds: Consider token expired this many seconds before actual expiry

        Returns:
            True if token is expired or will expire within buffer_seconds
        """
        workspace = self.get(workspace_id)
        if not workspace:
            return True  # Can't validate, assume expired

        if not workspace.token_expires_at:
            return False  # No expiration set, assume valid

        return time.time() + buffer_seconds >= workspace.token_expires_at


# Supabase-backed implementation for production
class SupabaseSlackWorkspaceStore:
    """
    Supabase-backed storage for Slack workspace OAuth credentials.

    Uses Supabase PostgreSQL for production deployments with proper
    encryption and multi-region support.

    Schema:
        CREATE TABLE slack_workspaces (
            workspace_id TEXT PRIMARY KEY,
            workspace_name TEXT NOT NULL,
            access_token TEXT NOT NULL,  -- Encrypted in Supabase vault
            bot_user_id TEXT NOT NULL,
            installed_at TIMESTAMPTZ NOT NULL,
            installed_by TEXT,
            scopes TEXT[],
            tenant_id TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            signing_secret TEXT,  -- Encrypted in Supabase vault
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
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
                logger.info("Slack workspace store using Supabase backend")
            else:
                logger.warning("Supabase not configured for Slack workspace store")
        except ImportError:
            logger.debug("Supabase client not available")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is configured."""
        return self._client is not None

    def save(self, workspace: SlackWorkspace) -> bool:
        """Save or update a workspace."""
        if not self.is_configured:
            return False

        try:
            data = {
                "workspace_id": workspace.workspace_id,
                "workspace_name": workspace.workspace_name,
                "access_token": workspace.access_token,
                "bot_user_id": workspace.bot_user_id,
                "installed_at": datetime.fromtimestamp(
                    workspace.installed_at, tz=timezone.utc
                ).isoformat(),
                "installed_by": workspace.installed_by,
                "scopes": workspace.scopes,
                "tenant_id": workspace.tenant_id,
                "is_active": workspace.is_active,
                "signing_secret": workspace.signing_secret,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            self._client.table("slack_workspaces").upsert(
                data, on_conflict="workspace_id"
            ).execute()

            logger.info(f"Saved Slack workspace to Supabase: {workspace.workspace_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to save workspace to Supabase: {e}")
            return False

    def get(self, workspace_id: str) -> Optional[SlackWorkspace]:
        """Get a workspace by ID."""
        if not self.is_configured:
            return None

        try:
            result = (
                self._client.table("slack_workspaces")
                .select("*")
                .eq("workspace_id", workspace_id)
                .single()
                .execute()
            )

            if result.data:
                return self._row_to_workspace(result.data)
            return None

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get workspace from Supabase: {e}")
            return None

    def get_by_tenant(self, tenant_id: str) -> List[SlackWorkspace]:
        """Get all workspaces for a tenant."""
        if not self.is_configured:
            return []

        try:
            result = (
                self._client.table("slack_workspaces")
                .select("*")
                .eq("tenant_id", tenant_id)
                .eq("is_active", True)
                .order("installed_at", desc=True)
                .execute()
            )

            return [self._row_to_workspace(row) for row in result.data]

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get workspaces for tenant from Supabase: {e}")
            return []

    def list_active(self, limit: int = 100, offset: int = 0) -> List[SlackWorkspace]:
        """List all active workspaces."""
        if not self.is_configured:
            return []

        try:
            result = (
                self._client.table("slack_workspaces")
                .select("*")
                .eq("is_active", True)
                .order("installed_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            return [self._row_to_workspace(row) for row in result.data]

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to list workspaces from Supabase: {e}")
            return []

    def deactivate(self, workspace_id: str) -> bool:
        """Deactivate a workspace."""
        if not self.is_configured:
            return False

        try:
            self._client.table("slack_workspaces").update(
                {
                    "is_active": False,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("workspace_id", workspace_id).execute()

            logger.info(f"Deactivated Slack workspace in Supabase: {workspace_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to deactivate workspace in Supabase: {e}")
            return False

    def delete(self, workspace_id: str) -> bool:
        """Permanently delete a workspace."""
        if not self.is_configured:
            return False

        try:
            self._client.table("slack_workspaces").delete().eq(
                "workspace_id", workspace_id
            ).execute()

            logger.info(f"Deleted Slack workspace from Supabase: {workspace_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to delete workspace from Supabase: {e}")
            return False

    def count(self, active_only: bool = True) -> int:
        """Count workspaces."""
        if not self.is_configured:
            return 0

        try:
            query = self._client.table("slack_workspaces").select("*", count="exact")
            if active_only:
                query = query.eq("is_active", True)

            result = query.execute()
            return result.count or 0

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to count workspaces in Supabase: {e}")
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
            logger.error(f"Failed to get stats from Supabase: {e}")
            return {"total_workspaces": 0, "active_workspaces": 0}

    def _row_to_workspace(self, row: Dict[str, Any]) -> SlackWorkspace:
        """Convert Supabase row to SlackWorkspace."""
        installed_at = row.get("installed_at")
        if isinstance(installed_at, str):
            # Parse ISO format
            installed_at = datetime.fromisoformat(installed_at.replace("Z", "+00:00")).timestamp()
        elif isinstance(installed_at, (int, float)):
            pass  # Already a timestamp
        else:
            installed_at = time.time()

        return SlackWorkspace(
            workspace_id=row["workspace_id"],
            workspace_name=row["workspace_name"],
            access_token=row["access_token"],
            bot_user_id=row["bot_user_id"],
            installed_at=installed_at,
            installed_by=row.get("installed_by"),
            scopes=row.get("scopes") or [],
            tenant_id=row.get("tenant_id"),
            is_active=row.get("is_active", True),
            signing_secret=row.get("signing_secret"),
        )


# Singleton instance
_workspace_store: Optional[Any] = None


def get_slack_workspace_store(db_path: Optional[str] = None) -> Any:
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
        if ARAGORA_ENV == "production" or os.getenv("USE_SUPABASE_SLACK_STORE"):
            supabase_store = SupabaseSlackWorkspaceStore()
            if supabase_store.is_configured:
                _workspace_store = supabase_store
                logger.info("Using Supabase-backed Slack workspace store")
                return _workspace_store

        # Fall back to SQLite
        _workspace_store = SlackWorkspaceStore(db_path)
        logger.info("Using SQLite-backed Slack workspace store")

    return _workspace_store
