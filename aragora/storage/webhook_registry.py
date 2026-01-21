"""
Webhook Registration Storage.

Provides durable storage for webhook registrations, replacing the in-memory
dict in handlers/webhooks.py. Survives server restarts.

Backends:
- SQLiteWebhookRegistry: Persisted, single-instance (default)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, cast

logger = logging.getLogger(__name__)


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook."""

    id: str
    url: str
    events: List[str]
    secret: str  # Used for HMAC signature
    active: bool = True
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Optional metadata
    name: Optional[str] = None
    description: Optional[str] = None

    # Delivery tracking
    last_delivery_at: Optional[float] = None
    last_delivery_status: Optional[int] = None
    delivery_count: int = 0
    failure_count: int = 0

    # Owner (for multi-tenant)
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None

    def to_dict(self, include_secret: bool = False) -> dict:
        """Convert to dict, optionally excluding secret."""
        result = asdict(self)
        if not include_secret:
            result.pop("secret", None)
        return result

    def matches_event(self, event_type: str) -> bool:
        """Check if this webhook should receive the given event."""
        if not self.active:
            return False
        # "*" means all events
        if "*" in self.events:
            return True
        return event_type in self.events


class SQLiteWebhookRegistry:
    """
    SQLite-backed webhook registration store.

    Provides durable storage for webhook configurations with the same
    interface as the in-memory WebhookStore in handlers/webhooks.py.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path | str):
        """
        Initialize SQLite webhook registry.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"SQLiteWebhookRegistry initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(
            """
            -- Webhook registrations table
            CREATE TABLE IF NOT EXISTS webhook_registrations (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                events TEXT NOT NULL,  -- JSON array
                secret TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                name TEXT,
                description TEXT,
                last_delivery_at REAL,
                last_delivery_status INTEGER,
                delivery_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                user_id TEXT,
                workspace_id TEXT
            );

            -- Indexes for common queries
            CREATE INDEX IF NOT EXISTS idx_webhook_user
                ON webhook_registrations(user_id);
            CREATE INDEX IF NOT EXISTS idx_webhook_workspace
                ON webhook_registrations(workspace_id);
            CREATE INDEX IF NOT EXISTS idx_webhook_active
                ON webhook_registrations(active);
            CREATE INDEX IF NOT EXISTS idx_webhook_created
                ON webhook_registrations(created_at);

            -- Schema version tracking
            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
        """
        )
        # Record schema version
        conn.execute(
            """INSERT OR REPLACE INTO _schema_versions (module, version)
               VALUES ('webhook_registry', ?)""",
            (self.SCHEMA_VERSION,),
        )
        conn.commit()
        conn.close()

    def _row_to_config(self, row: sqlite3.Row) -> WebhookConfig:
        """Convert database row to WebhookConfig."""
        return WebhookConfig(
            id=row["id"],
            url=row["url"],
            events=json.loads(row["events"]),
            secret=row["secret"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            name=row["name"],
            description=row["description"],
            last_delivery_at=row["last_delivery_at"],
            last_delivery_status=row["last_delivery_status"],
            delivery_count=row["delivery_count"] or 0,
            failure_count=row["failure_count"] or 0,
            user_id=row["user_id"],
            workspace_id=row["workspace_id"],
        )

    def register(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> WebhookConfig:
        """Register a new webhook."""
        webhook_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(32)
        now = time.time()

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO webhook_registrations
               (id, url, events, secret, active, created_at, updated_at,
                name, description, user_id, workspace_id)
               VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)""",
            (
                webhook_id,
                url,
                json.dumps(events),
                secret,
                now,
                now,
                name,
                description,
                user_id,
                workspace_id,
            ),
        )
        conn.commit()

        logger.info(f"Registered webhook {webhook_id} for events: {events}")

        return WebhookConfig(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            name=name,
            description=description,
            user_id=user_id,
            workspace_id=workspace_id,
            created_at=now,
            updated_at=now,
        )

    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT * FROM webhook_registrations WHERE id = ?",
            (webhook_id,),
        )
        row = cursor.fetchone()
        return self._row_to_config(row) if row else None

    def list(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        """List webhooks with optional filtering."""
        conn = self._get_conn()

        conditions = []
        params: List[Any] = []

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if active_only:
            conditions.append("active = 1")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM webhook_registrations WHERE {where_clause} ORDER BY created_at DESC"

        cursor = conn.execute(query, params)
        return [self._row_to_config(row) for row in cursor.fetchall()]

    # Alias for interface compatibility
    def list_all(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[WebhookConfig]:
        """List all webhooks (alias for list())."""
        return self.list(user_id=user_id, workspace_id=workspace_id)

    def delete(self, webhook_id: str) -> bool:
        """Delete webhook by ID."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM webhook_registrations WHERE id = ?",
            (webhook_id,),
        )
        conn.commit()

        if cursor.rowcount > 0:
            logger.info(f"Deleted webhook {webhook_id}")
            return True
        return False

    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[WebhookConfig]:
        """Update webhook configuration."""
        webhook = self.get(webhook_id)
        if not webhook:
            return None

        conn = self._get_conn()
        updates = ["updated_at = ?"]
        params: List[Any] = [time.time()]

        if url is not None:
            updates.append("url = ?")
            params.append(url)
        if events is not None:
            updates.append("events = ?")
            params.append(json.dumps(events))
        if active is not None:
            updates.append("active = ?")
            params.append(1 if active else 0)
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if description is not None:
            updates.append("description = ?")
            params.append(description)

        params.append(webhook_id)
        conn.execute(
            f"UPDATE webhook_registrations SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()

        return self.get(webhook_id)

    def record_delivery(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        """Record webhook delivery attempt."""
        conn = self._get_conn()
        now = time.time()

        if success:
            conn.execute(
                """UPDATE webhook_registrations SET
                   last_delivery_at = ?,
                   last_delivery_status = ?,
                   delivery_count = delivery_count + 1
                   WHERE id = ?""",
                (now, status_code, webhook_id),
            )
        else:
            conn.execute(
                """UPDATE webhook_registrations SET
                   last_delivery_at = ?,
                   last_delivery_status = ?,
                   delivery_count = delivery_count + 1,
                   failure_count = failure_count + 1
                   WHERE id = ?""",
                (now, status_code, webhook_id),
            )
        conn.commit()

    def get_for_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[WebhookConfig]:
        """Get all active webhooks that should receive the given event."""
        conn = self._get_conn()

        conditions = ["active = 1"]
        params: List[Any] = []

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)

        where_clause = " AND ".join(conditions)
        cursor = conn.execute(
            f"SELECT * FROM webhook_registrations WHERE {where_clause}",
            params,
        )

        webhooks = []
        for row in cursor.fetchall():
            config = self._row_to_config(row)
            if config.matches_event(event_type):
                webhooks.append(config)

        return webhooks

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# Global webhook registry instance
_webhook_registry: Optional[SQLiteWebhookRegistry] = None


def get_webhook_registry() -> SQLiteWebhookRegistry:
    """
    Get or create the webhook registration store.

    Uses environment variables to configure:
    - ARAGORA_DATA_DIR: Directory for SQLite database

    Returns:
        SQLiteWebhookRegistry instance
    """
    global _webhook_registry
    if _webhook_registry is not None:
        return _webhook_registry

    # Import DATA_DIR from config
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
        data_dir = Path(env_dir or ".nomic")

    _webhook_registry = SQLiteWebhookRegistry(data_dir / "webhook_registry.db")
    return _webhook_registry


def set_webhook_registry(registry: SQLiteWebhookRegistry) -> None:
    """Set custom webhook registry (for testing)."""
    global _webhook_registry
    _webhook_registry = registry


def reset_webhook_registry() -> None:
    """Reset the global webhook registry (for testing)."""
    global _webhook_registry
    _webhook_registry = None


__all__ = [
    "WebhookConfig",
    "SQLiteWebhookRegistry",
    "get_webhook_registry",
    "set_webhook_registry",
    "reset_webhook_registry",
]
