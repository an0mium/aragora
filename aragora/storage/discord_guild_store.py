"""
Discord Guild Storage for OAuth token management.

Stores guild credentials after OAuth installation for multi-guild support.
Tokens are encrypted at rest using AES-256-GCM when ARAGORA_ENCRYPTION_KEY is set.

Schema:
    CREATE TABLE discord_guilds (
        guild_id TEXT PRIMARY KEY,
        guild_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        refresh_token TEXT,
        bot_user_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        tenant_id TEXT,
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
DISCORD_GUILD_DB_PATH = os.environ.get(
    "DISCORD_GUILD_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "discord_guilds.db"),
)

# Encryption key for tokens (optional but recommended)
ENCRYPTION_KEY = os.environ.get("ARAGORA_ENCRYPTION_KEY", "")


@dataclass
class DiscordGuild:
    """Represents an installed Discord guild (server)."""

    guild_id: str  # Discord server ID (snowflake)
    guild_name: str
    access_token: str  # Bot access token
    bot_user_id: str  # Bot user ID
    installed_at: float  # Unix timestamp
    refresh_token: Optional[str] = None  # OAuth refresh token
    installed_by: Optional[str] = None  # User ID who installed
    scopes: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None  # Link to Aragora tenant
    is_active: bool = True
    expires_at: Optional[float] = None  # Token expiration timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes sensitive tokens)."""
        return {
            "guild_id": self.guild_id,
            "guild_name": self.guild_name,
            "bot_user_id": self.bot_user_id,
            "installed_at": self.installed_at,
            "installed_at_iso": datetime.fromtimestamp(
                self.installed_at, tz=timezone.utc
            ).isoformat(),
            "installed_by": self.installed_by,
            "scopes": self.scopes,
            "tenant_id": self.tenant_id,
            "is_active": self.is_active,
            "expires_at": self.expires_at,
            "expires_at_iso": (
                datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat()
                if self.expires_at
                else None
            ),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DiscordGuild":
        """Create from database row."""
        scopes_str = row["scopes"] or ""
        scopes = scopes_str.split(",") if scopes_str else []

        return cls(
            guild_id=row["guild_id"],
            guild_name=row["guild_name"],
            access_token=row["access_token"],
            refresh_token=row["refresh_token"],
            bot_user_id=row["bot_user_id"],
            installed_at=row["installed_at"],
            installed_by=row["installed_by"],
            scopes=scopes,
            tenant_id=row["tenant_id"],
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


class DiscordGuildStore:
    """
    Storage for Discord guild OAuth credentials.

    Supports SQLite backend with optional token encryption.
    Thread-safe for concurrent access.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS discord_guilds (
        guild_id TEXT PRIMARY KEY,
        guild_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        refresh_token TEXT,
        bot_user_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        tenant_id TEXT,
        is_active INTEGER DEFAULT 1,
        expires_at REAL
    );

    CREATE INDEX IF NOT EXISTS idx_discord_guilds_tenant
        ON discord_guilds(tenant_id);

    CREATE INDEX IF NOT EXISTS idx_discord_guilds_active
        ON discord_guilds(is_active);

    CREATE INDEX IF NOT EXISTS idx_discord_guilds_expires
        ON discord_guilds(expires_at);
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the guild store.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path or DISCORD_GUILD_DB_PATH
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
        except Exception as e:
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
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            return encrypted

    def save(self, guild: DiscordGuild) -> bool:
        """Save or update a guild.

        Args:
            guild: Guild to save

        Returns:
            True if saved successfully
        """
        conn = self._get_connection()
        try:
            encrypted_access = self._encrypt_token(guild.access_token)
            encrypted_refresh = (
                self._encrypt_token(guild.refresh_token) if guild.refresh_token else None
            )
            scopes_str = ",".join(guild.scopes)

            conn.execute(
                """
                INSERT OR REPLACE INTO discord_guilds
                (guild_id, guild_name, access_token, refresh_token, bot_user_id,
                 installed_at, installed_by, scopes, tenant_id, is_active, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild.guild_id,
                    guild.guild_name,
                    encrypted_access,
                    encrypted_refresh,
                    guild.bot_user_id,
                    guild.installed_at,
                    guild.installed_by,
                    scopes_str,
                    guild.tenant_id,
                    1 if guild.is_active else 0,
                    guild.expires_at,
                ),
            )
            conn.commit()
            logger.info(f"Saved Discord guild: {guild.guild_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save guild: {e}")
            return False

    def get(self, guild_id: str) -> Optional[DiscordGuild]:
        """Get a guild by ID.

        Args:
            guild_id: Discord guild ID

        Returns:
            Guild or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM discord_guilds WHERE guild_id = ?",
                (guild_id,),
            )
            row = cursor.fetchone()

            if row:
                guild = DiscordGuild.from_row(row)
                guild.access_token = self._decrypt_token(guild.access_token)
                if guild.refresh_token:
                    guild.refresh_token = self._decrypt_token(guild.refresh_token)
                return guild

            return None

        except Exception as e:
            logger.error(f"Failed to get guild {guild_id}: {e}")
            return None

    def get_by_tenant(self, tenant_id: str) -> List[DiscordGuild]:
        """Get all guilds for an Aragora tenant.

        Args:
            tenant_id: Aragora tenant ID

        Returns:
            List of guilds
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM discord_guilds
                WHERE tenant_id = ? AND is_active = 1
                ORDER BY installed_at DESC
                """,
                (tenant_id,),
            )

            guilds = []
            for row in cursor.fetchall():
                guild = DiscordGuild.from_row(row)
                guild.access_token = self._decrypt_token(guild.access_token)
                if guild.refresh_token:
                    guild.refresh_token = self._decrypt_token(guild.refresh_token)
                guilds.append(guild)

            return guilds

        except Exception as e:
            logger.error(f"Failed to get guilds for tenant {tenant_id}: {e}")
            return []

    def list_active(self, limit: int = 100, offset: int = 0) -> List[DiscordGuild]:
        """List all active guilds.

        Args:
            limit: Maximum number of guilds to return
            offset: Pagination offset

        Returns:
            List of active guilds
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM discord_guilds
                WHERE is_active = 1
                ORDER BY installed_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

            guilds = []
            for row in cursor.fetchall():
                guild = DiscordGuild.from_row(row)
                guild.access_token = self._decrypt_token(guild.access_token)
                if guild.refresh_token:
                    guild.refresh_token = self._decrypt_token(guild.refresh_token)
                guilds.append(guild)

            return guilds

        except Exception as e:
            logger.error(f"Failed to list guilds: {e}")
            return []

    def list_expiring(self, within_seconds: int = 3600) -> List[DiscordGuild]:
        """List guilds with tokens expiring soon.

        Args:
            within_seconds: Time window in seconds (default 1 hour)

        Returns:
            List of guilds with expiring tokens
        """
        import time

        conn = self._get_connection()
        try:
            cutoff = time.time() + within_seconds
            cursor = conn.execute(
                """
                SELECT * FROM discord_guilds
                WHERE is_active = 1 AND expires_at IS NOT NULL AND expires_at < ?
                ORDER BY expires_at ASC
                """,
                (cutoff,),
            )

            guilds = []
            for row in cursor.fetchall():
                guild = DiscordGuild.from_row(row)
                guild.access_token = self._decrypt_token(guild.access_token)
                if guild.refresh_token:
                    guild.refresh_token = self._decrypt_token(guild.refresh_token)
                guilds.append(guild)

            return guilds

        except Exception as e:
            logger.error(f"Failed to list expiring guilds: {e}")
            return []

    def update_tokens(
        self,
        guild_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> bool:
        """Update tokens for a guild (after refresh).

        Args:
            guild_id: Discord guild ID
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
                    UPDATE discord_guilds
                    SET access_token = ?, refresh_token = ?, expires_at = ?
                    WHERE guild_id = ?
                    """,
                    (encrypted_access, encrypted_refresh, expires_at, guild_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE discord_guilds
                    SET access_token = ?, expires_at = ?
                    WHERE guild_id = ?
                    """,
                    (encrypted_access, expires_at, guild_id),
                )

            conn.commit()
            logger.info(f"Updated tokens for Discord guild: {guild_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update tokens for guild {guild_id}: {e}")
            return False

    def deactivate(self, guild_id: str) -> bool:
        """Deactivate a guild (on uninstall).

        Args:
            guild_id: Discord guild ID

        Returns:
            True if deactivated successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE discord_guilds SET is_active = 0 WHERE guild_id = ?",
                (guild_id,),
            )
            conn.commit()
            logger.info(f"Deactivated Discord guild: {guild_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deactivate guild {guild_id}: {e}")
            return False

    def delete(self, guild_id: str) -> bool:
        """Permanently delete a guild.

        Args:
            guild_id: Discord guild ID

        Returns:
            True if deleted successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM discord_guilds WHERE guild_id = ?",
                (guild_id,),
            )
            conn.commit()
            logger.info(f"Deleted Discord guild: {guild_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete guild {guild_id}: {e}")
            return False

    def count(self, active_only: bool = True) -> int:
        """Count guilds.

        Args:
            active_only: Only count active guilds

        Returns:
            Number of guilds
        """
        conn = self._get_connection()
        try:
            if active_only:
                cursor = conn.execute("SELECT COUNT(*) FROM discord_guilds WHERE is_active = 1")
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM discord_guilds")

            return cursor.fetchone()[0]

        except Exception as e:
            logger.error(f"Failed to count guilds: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get guild statistics.

        Returns:
            Statistics dictionary
        """
        import time

        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM discord_guilds").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM discord_guilds WHERE is_active = 1"
            ).fetchone()[0]

            # Count expiring tokens (within 1 hour)
            cutoff = time.time() + 3600
            expiring = conn.execute(
                "SELECT COUNT(*) FROM discord_guilds WHERE is_active = 1 AND expires_at < ?",
                (cutoff,),
            ).fetchone()[0]

            return {
                "total_guilds": total,
                "active_guilds": active,
                "inactive_guilds": total - active,
                "expiring_tokens": expiring,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_guilds": 0, "active_guilds": 0}


# Singleton instance
_guild_store: Optional[DiscordGuildStore] = None


def get_discord_guild_store(db_path: Optional[str] = None) -> DiscordGuildStore:
    """Get or create the guild store singleton.

    Args:
        db_path: Optional path to database file

    Returns:
        DiscordGuildStore instance
    """
    global _guild_store
    if _guild_store is None:
        _guild_store = DiscordGuildStore(db_path)
    return _guild_store
