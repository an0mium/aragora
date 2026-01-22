"""
Notification Configuration Store.

Provides per-organization persistent storage for email and telegram notification
configurations. Supports multi-tenancy with tenant-scoped data isolation.

Backends:
- SQLiteNotificationConfigStore: Persisted, single-instance (default)
- In-memory fallback when database unavailable

Usage:
    from aragora.storage.notification_config_store import get_notification_config_store

    store = get_notification_config_store()
    await store.save_email_config(org_id, config)
    config = await store.get_email_config(org_id)
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.storage.backends import (
    POSTGRESQL_AVAILABLE,
    DatabaseBackend,
    PostgreSQLBackend,
    SQLiteBackend,
)

logger = logging.getLogger(__name__)

# Import encryption (optional - graceful degradation if not available)
# Use type: ignore to handle conditional definitions cleanly
try:
    from aragora.security.encryption import (
        get_encryption_service,
        is_encryption_required,
        EncryptionError,
        CRYPTO_AVAILABLE,
    )
except ImportError:
    CRYPTO_AVAILABLE = False  # type: ignore[misc]

    class EncryptionError(Exception):  # type: ignore[no-redef]
        """Fallback exception when security module unavailable."""

        def __init__(self, operation: str, reason: str, store: str = ""):
            self.operation = operation
            self.reason = reason
            self.store = store
            super().__init__(f"Encryption {operation} failed: {reason}")

    def get_encryption_service() -> Any:  # type: ignore[misc]
        raise RuntimeError("Encryption not available")

    def is_encryption_required() -> bool:
        """Fallback when security module unavailable."""
        if os.environ.get("ARAGORA_ENCRYPTION_REQUIRED", "").lower() in ("true", "1", "yes"):
            return True
        if os.environ.get("ARAGORA_ENV") == "production":
            return True
        return False


# Sensitive keys that should be encrypted
SENSITIVE_KEYS = frozenset(
    [
        "smtp_password",
        "sendgrid_api_key",
        "ses_secret_access_key",
        "bot_token",
    ]
)


def _encrypt_config(
    config: Dict[str, Any],
    org_id: str,
    config_type: str,
) -> Dict[str, Any]:
    """Encrypt sensitive keys in config dict before storage."""
    if not config:
        return config

    keys_to_encrypt = [k for k in SENSITIVE_KEYS if k in config and config[k]]
    if not keys_to_encrypt:
        return config

    if not CRYPTO_AVAILABLE:
        if is_encryption_required():
            raise EncryptionError(
                "encrypt", "cryptography library not available", "notification_config_store"
            )
        return config

    try:
        service = get_encryption_service()
        aad = f"{org_id}:{config_type}"
        encrypted = service.encrypt_fields(config, keys_to_encrypt, aad)
        logger.debug(f"Encrypted {len(keys_to_encrypt)} sensitive fields for {config_type}")
        return encrypted
    except Exception as e:
        if is_encryption_required():
            raise EncryptionError("encrypt", str(e), "notification_config_store") from e
        logger.warning(f"Encryption unavailable, storing unencrypted: {e}")
        return config


def _decrypt_config(
    config: Dict[str, Any],
    org_id: str,
    config_type: str,
) -> Dict[str, Any]:
    """Decrypt sensitive keys in config dict."""
    if not CRYPTO_AVAILABLE or not config:
        return config

    encrypted_keys = [
        k
        for k in SENSITIVE_KEYS
        if k in config and isinstance(config.get(k), dict) and config[k].get("_encrypted")
    ]
    if not encrypted_keys:
        return config

    try:
        service = get_encryption_service()
        aad = f"{org_id}:{config_type}"
        decrypted = service.decrypt_fields(config, encrypted_keys, aad)
        logger.debug(f"Decrypted {len(encrypted_keys)} fields for {config_type}")
        return decrypted
    except Exception as e:
        logger.warning(f"Decryption failed for {config_type}: {e}")
        return config


@dataclass
class StoredEmailConfig:
    """Stored email configuration for an organization."""

    org_id: str
    provider: str = "smtp"
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    use_tls: bool = True
    use_ssl: bool = False
    sendgrid_api_key: str = ""
    ses_region: str = "us-east-1"
    ses_access_key_id: str = ""
    ses_secret_access_key: str = ""
    from_email: str = "debates@aragora.ai"
    from_name: str = "Aragora Debates"
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True
    enable_digest: bool = True
    digest_frequency: str = "daily"
    min_consensus_confidence: float = 0.7
    max_emails_per_hour: int = 50
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredEmailConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StoredTelegramConfig:
    """Stored telegram configuration for an organization."""

    org_id: str
    bot_token: str = ""
    chat_id: str = ""
    notify_on_consensus: bool = True
    notify_on_debate_end: bool = True
    notify_on_error: bool = True
    min_consensus_confidence: float = 0.7
    max_messages_per_minute: int = 20
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredTelegramConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class StoredEmailRecipient:
    """Stored email recipient for an organization."""

    org_id: str
    email: str
    name: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredEmailRecipient":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class NotificationConfigStore:
    """
    Database-backed notification config store with per-org tenant isolation.

    Supports SQLite (default) and PostgreSQL backends.
    All configurations are scoped to org_id to ensure multi-tenant data isolation.
    Sensitive fields (passwords, tokens) are encrypted at rest.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        backend: Optional[str] = None,
        database_url: Optional[str] = None,
    ):
        """Initialize the store with optional database path."""
        if db_path is None:
            data_dir = Path(os.environ.get("ARAGORA_DATA_DIR", ".aragora"))
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "notification_config.db")

        self._db_path = db_path
        self._local = threading.local()

        # Determine backend type
        env_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_DATABASE_URL")
        actual_url = database_url or env_url

        if backend is None:
            env_backend = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()
            backend = "postgresql" if (actual_url and env_backend == "postgresql") else "sqlite"

        self.backend_type = backend
        self._backend: Optional[DatabaseBackend] = None

        # Initialize backend
        if backend == "postgresql":
            if not actual_url:
                raise ValueError("PostgreSQL backend requires DATABASE_URL")
            if not POSTGRESQL_AVAILABLE:
                raise ImportError("psycopg2 required for PostgreSQL")
            self._backend = PostgreSQLBackend(actual_url)
            logger.info("NotificationConfigStore using PostgreSQL backend")
        else:
            self._backend = SQLiteBackend(db_path)
            logger.info(f"NotificationConfigStore using SQLite backend: {db_path}")

        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection (legacy)."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        if self._backend is not None:
            # Email configurations table
            self._backend.execute_write("""
                CREATE TABLE IF NOT EXISTS email_configs (
                    org_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Telegram configurations table
            self._backend.execute_write("""
                CREATE TABLE IF NOT EXISTS telegram_configs (
                    org_id TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Email recipients table - use SERIAL for PostgreSQL
            if self.backend_type == "postgresql":
                self._backend.execute_write("""
                    CREATE TABLE IF NOT EXISTS email_recipients (
                        id SERIAL PRIMARY KEY,
                        org_id TEXT NOT NULL,
                        email TEXT NOT NULL,
                        name TEXT,
                        preferences_json TEXT,
                        created_at REAL NOT NULL,
                        UNIQUE(org_id, email)
                    )
                """)
            else:
                self._backend.execute_write("""
                    CREATE TABLE IF NOT EXISTS email_recipients (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        org_id TEXT NOT NULL,
                        email TEXT NOT NULL,
                        name TEXT,
                        preferences_json TEXT,
                        created_at REAL NOT NULL,
                        UNIQUE(org_id, email)
                    )
                """)

            # Create indexes
            try:
                self._backend.execute_write(
                    "CREATE INDEX IF NOT EXISTS idx_recipients_org ON email_recipients(org_id)"
                )
            except Exception as e:
                logger.debug(f"Index creation skipped: {e}")

            logger.info(f"NotificationConfigStore initialized with {self.backend_type} backend")
            return

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()

        # Email configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS email_configs (
                org_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # Telegram configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telegram_configs (
                org_id TEXT PRIMARY KEY,
                config_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # Email recipients table (multiple per org)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS email_recipients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id TEXT NOT NULL,
                email TEXT NOT NULL,
                name TEXT,
                preferences_json TEXT,
                created_at REAL NOT NULL,
                UNIQUE(org_id, email)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recipients_org ON email_recipients(org_id)")

        conn.commit()
        logger.info(f"NotificationConfigStore initialized at {self._db_path}")

    # =========================================================================
    # Email Config Operations
    # =========================================================================

    async def save_email_config(self, config: StoredEmailConfig) -> None:
        """Save email configuration for an organization."""
        config_dict = config.to_dict()
        config_dict["updated_at"] = time.time()

        # Encrypt sensitive fields
        encrypted = _encrypt_config(config_dict, config.org_id, "email")

        params = (config.org_id, json.dumps(encrypted), config.created_at, encrypted["updated_at"])

        if self._backend is not None:
            self._backend.execute_write(
                """
                INSERT INTO email_configs (org_id, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(org_id) DO UPDATE SET
                    config_json = EXCLUDED.config_json,
                    updated_at = EXCLUDED.updated_at
                """,
                params,
            )
            logger.debug(f"Saved email config for org {config.org_id}")
            return

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO email_configs (org_id, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(org_id) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at = excluded.updated_at
            """,
            params,
        )
        conn.commit()
        logger.debug(f"Saved email config for org {config.org_id}")

    async def get_email_config(self, org_id: str) -> Optional[StoredEmailConfig]:
        """Get email configuration for an organization."""
        if self._backend is not None:
            row = self._backend.fetch_one(
                "SELECT config_json FROM email_configs WHERE org_id = ?",
                (org_id,),
            )
            if not row:
                return None
            config_dict = json.loads(row[0])
            decrypted = _decrypt_config(config_dict, org_id, "email")
            return StoredEmailConfig.from_dict(decrypted)

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT config_json FROM email_configs WHERE org_id = ?", (org_id,))
        row = cursor.fetchone()

        if not row:
            return None

        config_dict = json.loads(row["config_json"])
        # Decrypt sensitive fields
        decrypted = _decrypt_config(config_dict, org_id, "email")
        return StoredEmailConfig.from_dict(decrypted)

    async def delete_email_config(self, org_id: str) -> bool:
        """Delete email configuration for an organization."""
        if self._backend is not None:
            # Count before delete to determine if anything was deleted
            row = self._backend.fetch_one(
                "SELECT COUNT(*) FROM email_configs WHERE org_id = ?",
                (org_id,),
            )
            count_before = row[0] if row else 0
            if count_before > 0:
                self._backend.execute_write(
                    "DELETE FROM email_configs WHERE org_id = ?",
                    (org_id,),
                )
            return count_before > 0

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM email_configs WHERE org_id = ?", (org_id,))
        conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # Telegram Config Operations
    # =========================================================================

    async def save_telegram_config(self, config: StoredTelegramConfig) -> None:
        """Save telegram configuration for an organization."""
        config_dict = config.to_dict()
        config_dict["updated_at"] = time.time()

        # Encrypt sensitive fields
        encrypted = _encrypt_config(config_dict, config.org_id, "telegram")

        params = (config.org_id, json.dumps(encrypted), config.created_at, encrypted["updated_at"])

        if self._backend is not None:
            self._backend.execute_write(
                """
                INSERT INTO telegram_configs (org_id, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(org_id) DO UPDATE SET
                    config_json = EXCLUDED.config_json,
                    updated_at = EXCLUDED.updated_at
                """,
                params,
            )
            logger.debug(f"Saved telegram config for org {config.org_id}")
            return

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO telegram_configs (org_id, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(org_id) DO UPDATE SET
                config_json = excluded.config_json,
                updated_at = excluded.updated_at
            """,
            params,
        )
        conn.commit()
        logger.debug(f"Saved telegram config for org {config.org_id}")

    async def get_telegram_config(self, org_id: str) -> Optional[StoredTelegramConfig]:
        """Get telegram configuration for an organization."""
        if self._backend is not None:
            row = self._backend.fetch_one(
                "SELECT config_json FROM telegram_configs WHERE org_id = ?",
                (org_id,),
            )
            if not row:
                return None
            config_dict = json.loads(row[0])
            decrypted = _decrypt_config(config_dict, org_id, "telegram")
            return StoredTelegramConfig.from_dict(decrypted)

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT config_json FROM telegram_configs WHERE org_id = ?", (org_id,))
        row = cursor.fetchone()

        if not row:
            return None

        config_dict = json.loads(row["config_json"])
        # Decrypt sensitive fields
        decrypted = _decrypt_config(config_dict, org_id, "telegram")
        return StoredTelegramConfig.from_dict(decrypted)

    async def delete_telegram_config(self, org_id: str) -> bool:
        """Delete telegram configuration for an organization."""
        if self._backend is not None:
            # Count before delete to determine if anything was deleted
            row = self._backend.fetch_one(
                "SELECT COUNT(*) FROM telegram_configs WHERE org_id = ?",
                (org_id,),
            )
            count_before = row[0] if row else 0
            if count_before > 0:
                self._backend.execute_write(
                    "DELETE FROM telegram_configs WHERE org_id = ?",
                    (org_id,),
                )
            return count_before > 0

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM telegram_configs WHERE org_id = ?", (org_id,))
        conn.commit()
        return cursor.rowcount > 0

    # =========================================================================
    # Email Recipients Operations
    # =========================================================================

    async def add_recipient(self, recipient: StoredEmailRecipient) -> None:
        """Add an email recipient for an organization."""
        params = (
            recipient.org_id,
            recipient.email,
            recipient.name,
            json.dumps(recipient.preferences),
            recipient.created_at,
        )

        if self._backend is not None:
            self._backend.execute_write(
                """
                INSERT INTO email_recipients (org_id, email, name, preferences_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(org_id, email) DO UPDATE SET
                    name = EXCLUDED.name,
                    preferences_json = EXCLUDED.preferences_json
                """,
                params,
            )
            logger.debug(f"Added recipient {recipient.email} for org {recipient.org_id}")
            return

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO email_recipients (org_id, email, name, preferences_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(org_id, email) DO UPDATE SET
                name = excluded.name,
                preferences_json = excluded.preferences_json
            """,
            params,
        )
        conn.commit()
        logger.debug(f"Added recipient {recipient.email} for org {recipient.org_id}")

    async def get_recipients(self, org_id: str) -> List[StoredEmailRecipient]:
        """Get all email recipients for an organization."""
        if self._backend is not None:
            rows = self._backend.fetch_all(
                "SELECT org_id, email, name, preferences_json, created_at FROM email_recipients WHERE org_id = ?",
                (org_id,),
            )
            recipients = []
            for row in rows:
                recipients.append(
                    StoredEmailRecipient(
                        org_id=row[0],
                        email=row[1],
                        name=row[2],
                        preferences=json.loads(row[3]) if row[3] else {},
                        created_at=row[4],
                    )
                )
            return recipients

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT org_id, email, name, preferences_json, created_at FROM email_recipients WHERE org_id = ?",
            (org_id,),
        )
        rows = cursor.fetchall()

        recipients = []
        for row in rows:
            recipients.append(
                StoredEmailRecipient(
                    org_id=row["org_id"],  # type: ignore[call-overload]
                    email=row["email"],  # type: ignore[call-overload]
                    name=row["name"],  # type: ignore[call-overload]
                    preferences=json.loads(row["preferences_json"])  # type: ignore[call-overload]
                    if row["preferences_json"]  # type: ignore[call-overload]
                    else {},
                    created_at=row["created_at"],  # type: ignore[call-overload]
                )
            )
        return recipients

    async def remove_recipient(self, org_id: str, email: str) -> bool:
        """Remove an email recipient for an organization."""
        if self._backend is not None:
            # Count before delete to determine if anything was deleted
            row = self._backend.fetch_one(
                "SELECT COUNT(*) FROM email_recipients WHERE org_id = ? AND email = ?",
                (org_id, email),
            )
            count_before = row[0] if row else 0
            if count_before > 0:
                self._backend.execute_write(
                    "DELETE FROM email_recipients WHERE org_id = ? AND email = ?",
                    (org_id, email),
                )
            return count_before > 0

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM email_recipients WHERE org_id = ? AND email = ?",
            (org_id, email),
        )
        conn.commit()
        return cursor.rowcount > 0

    async def clear_recipients(self, org_id: str) -> int:
        """Clear all recipients for an organization."""
        if self._backend is not None:
            # Count before delete to return the count
            row = self._backend.fetch_one(
                "SELECT COUNT(*) FROM email_recipients WHERE org_id = ?",
                (org_id,),
            )
            count = row[0] if row else 0
            if count > 0:
                self._backend.execute_write(
                    "DELETE FROM email_recipients WHERE org_id = ?",
                    (org_id,),
                )
            return count

        # Legacy SQLite path
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM email_recipients WHERE org_id = ?", (org_id,))
        conn.commit()
        return cursor.rowcount

    def close(self) -> None:
        """Close database connection."""
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        elif hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


# =============================================================================
# Singleton Instance
# =============================================================================

_notification_config_store: Optional[NotificationConfigStore] = None
_store_lock = threading.Lock()


def get_notification_config_store(
    db_path: Optional[str] = None,
    backend: Optional[str] = None,
    database_url: Optional[str] = None,
) -> NotificationConfigStore:
    """Get the singleton notification config store instance.

    Args:
        db_path: Path to SQLite database file (used when backend="sqlite")
        backend: Database backend ("sqlite" or "postgresql")
        database_url: PostgreSQL connection URL

    Returns:
        Configured NotificationConfigStore instance
    """
    global _notification_config_store
    if _notification_config_store is None:
        with _store_lock:
            if _notification_config_store is None:
                _notification_config_store = NotificationConfigStore(
                    db_path=db_path,
                    backend=backend,
                    database_url=database_url,
                )
    return _notification_config_store


def reset_notification_config_store() -> None:
    """Reset the singleton store (for testing)."""
    global _notification_config_store
    with _store_lock:
        if _notification_config_store is not None:
            _notification_config_store.close()
            _notification_config_store = None


__all__ = [
    "NotificationConfigStore",
    "StoredEmailConfig",
    "StoredTelegramConfig",
    "StoredEmailRecipient",
    "get_notification_config_store",
    "reset_notification_config_store",
]
