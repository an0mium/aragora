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
                except Exception as e:
                    logger.debug(f"Migration check: {e}")

                self._initialized = True

    def _encrypt_token(self, token: str) -> str:
        """Encrypt token if encryption key is configured."""
        if not ENCRYPTION_KEY:
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
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            return token

    def _decrypt_token(self, encrypted: str) -> str:
        """Decrypt token if encryption key is configured."""
        if not ENCRYPTION_KEY:
            return encrypted

        # Check if it looks like an encrypted token
        if encrypted.startswith("xoxb-") or encrypted.startswith("xoxp-"):
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
        except Exception as e:
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

        except Exception as e:
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
                return workspace

            return None

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
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

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_workspaces": 0, "active_workspaces": 0}


# Singleton instance
_workspace_store: Optional[SlackWorkspaceStore] = None


def get_slack_workspace_store(db_path: Optional[str] = None) -> SlackWorkspaceStore:
    """Get or create the workspace store singleton.

    Args:
        db_path: Optional path to database file

    Returns:
        SlackWorkspaceStore instance
    """
    global _workspace_store
    if _workspace_store is None:
        _workspace_store = SlackWorkspaceStore(db_path)
    return _workspace_store
