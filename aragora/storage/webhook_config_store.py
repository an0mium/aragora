"""
Webhook Configuration Storage.

Provides persistent storage for webhook configurations (URLs, events, secrets).
Survives server restarts and supports multi-instance deployments.

Backends:
- InMemoryWebhookConfigStore: Fast, single-instance only (for testing)
- SQLiteWebhookConfigStore: Persisted, single-instance (default)
- RedisWebhookConfigStore: Distributed, multi-instance (with SQLite fallback)

Usage:
    from aragora.storage.webhook_config_store import get_webhook_config_store

    store = get_webhook_config_store()
    webhook = await store.register(url="https://...", events=["debate_end"])
    webhook = await store.get(webhook_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, cast

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)


# Events that can trigger webhooks
WEBHOOK_EVENTS: Set[str] = {
    "debate_start",
    "debate_end",
    "consensus",
    "round_start",
    "agent_message",
    "vote",
    "insight_extracted",
    "memory_stored",
    "memory_retrieved",
    "claim_verification_result",
    "formal_verification_result",
    "gauntlet_complete",
    "gauntlet_verdict",
    "receipt_ready",
    "receipt_exported",
    "graph_branch_created",
    "graph_branch_merged",
    "genesis_evolution",
    "breakpoint",
    "breakpoint_resolved",
    "agent_elo_updated",
    "knowledge_indexed",
    "knowledge_queried",
    "mound_updated",
    "calibration_update",
    "evidence_found",
    "agent_calibration_changed",
    "agent_fallback_triggered",
    "explanation_ready",
}


@dataclass
class WebhookConfig:
    """Configuration for a registered webhook."""

    id: str
    url: str
    events: List[str]
    secret: str
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

    def to_json(self) -> str:
        """Serialize to JSON for storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "WebhookConfig":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_row(cls, row: tuple) -> "WebhookConfig":
        """Create from database row."""
        return cls(
            id=row[0],
            url=row[1],
            events=json.loads(row[2]) if row[2] else [],
            secret=row[3],
            active=bool(row[4]),
            created_at=row[5] or time.time(),
            updated_at=row[6] or time.time(),
            name=row[7],
            description=row[8],
            last_delivery_at=row[9],
            last_delivery_status=row[10],
            delivery_count=row[11] or 0,
            failure_count=row[12] or 0,
            user_id=row[13],
            workspace_id=row[14],
        )

    def matches_event(self, event_type: str) -> bool:
        """Check if this webhook should receive the given event."""
        if not self.active:
            return False
        if "*" in self.events:
            return event_type in WEBHOOK_EVENTS
        return event_type in self.events


class WebhookConfigStoreBackend(ABC):
    """Abstract base for webhook configuration storage backends."""

    @abstractmethod
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
        pass

    @abstractmethod
    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID."""
        pass

    @abstractmethod
    def list(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        """List webhooks with optional filtering."""
        pass

    @abstractmethod
    def delete(self, webhook_id: str) -> bool:
        """Delete webhook by ID."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def record_delivery(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        """Record webhook delivery attempt."""
        pass

    @abstractmethod
    def get_for_event(self, event_type: str) -> List[WebhookConfig]:
        """Get all active webhooks that should receive the given event."""
        pass

    def close(self) -> None:
        """Close connections (optional to implement)."""
        pass


class InMemoryWebhookConfigStore(WebhookConfigStoreBackend):
    """
    Thread-safe in-memory webhook config store.

    Fast but not shared across restarts. Suitable for development/testing.
    """

    def __init__(self) -> None:
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._lock = threading.RLock()

    def register(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> WebhookConfig:
        webhook_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(32)

        webhook = WebhookConfig(
            id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            name=name,
            description=description,
            user_id=user_id,
            workspace_id=workspace_id,
        )

        with self._lock:
            self._webhooks[webhook_id] = webhook

        logger.info(f"Registered webhook {webhook_id} for events: {events}")
        return webhook

    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        with self._lock:
            return self._webhooks.get(webhook_id)

    def list(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        with self._lock:
            webhooks = list(self._webhooks.values())

        if user_id:
            webhooks = [w for w in webhooks if w.user_id == user_id]
        if workspace_id:
            webhooks = [w for w in webhooks if w.workspace_id == workspace_id]
        if active_only:
            webhooks = [w for w in webhooks if w.active]

        return sorted(webhooks, key=lambda w: w.created_at, reverse=True)

    def delete(self, webhook_id: str) -> bool:
        with self._lock:
            if webhook_id in self._webhooks:
                del self._webhooks[webhook_id]
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
        with self._lock:
            webhook = self._webhooks.get(webhook_id)
            if not webhook:
                return None

            if url is not None:
                webhook.url = url
            if events is not None:
                webhook.events = events
            if active is not None:
                webhook.active = active
            if name is not None:
                webhook.name = name
            if description is not None:
                webhook.description = description

            webhook.updated_at = time.time()
            return webhook

    def record_delivery(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        with self._lock:
            webhook = self._webhooks.get(webhook_id)
            if webhook:
                webhook.last_delivery_at = time.time()
                webhook.last_delivery_status = status_code
                webhook.delivery_count += 1
                if not success:
                    webhook.failure_count += 1

    def get_for_event(self, event_type: str) -> List[WebhookConfig]:
        with self._lock:
            return [w for w in self._webhooks.values() if w.matches_event(event_type)]

    def clear(self) -> None:
        """Clear all entries (for testing)."""
        with self._lock:
            self._webhooks.clear()


class SQLiteWebhookConfigStore(WebhookConfigStoreBackend):
    """
    SQLite-backed webhook config store.

    Persisted to disk, survives restarts. Suitable for single-instance
    production deployments.
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()
        logger.info(f"SQLiteWebhookConfigStore initialized: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return cast(sqlite3.Connection, self._local.conn)

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS webhook_configs (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                events_json TEXT NOT NULL,
                secret TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
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
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_webhook_configs_user ON webhook_configs(user_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_webhook_configs_workspace ON webhook_configs(workspace_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_webhook_configs_active ON webhook_configs(active)"
        )
        conn.commit()
        conn.close()

    def register(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> WebhookConfig:
        webhook_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(32)
        now = time.time()

        webhook = WebhookConfig(
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

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO webhook_configs
               (id, url, events_json, secret, active, created_at, updated_at,
                name, description, last_delivery_at, last_delivery_status,
                delivery_count, failure_count, user_id, workspace_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                webhook.id,
                webhook.url,
                json.dumps(webhook.events),
                webhook.secret,
                int(webhook.active),
                webhook.created_at,
                webhook.updated_at,
                webhook.name,
                webhook.description,
                webhook.last_delivery_at,
                webhook.last_delivery_status,
                webhook.delivery_count,
                webhook.failure_count,
                webhook.user_id,
                webhook.workspace_id,
            ),
        )
        conn.commit()
        logger.info(f"Registered webhook {webhook_id} for events: {events}")
        return webhook

    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT id, url, events_json, secret, active, created_at, updated_at,
                      name, description, last_delivery_at, last_delivery_status,
                      delivery_count, failure_count, user_id, workspace_id
               FROM webhook_configs WHERE id = ?""",
            (webhook_id,),
        )
        row = cursor.fetchone()
        if row:
            return WebhookConfig.from_row(row)
        return None

    def list(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        conn = self._get_conn()

        query = """SELECT id, url, events_json, secret, active, created_at, updated_at,
                          name, description, last_delivery_at, last_delivery_status,
                          delivery_count, failure_count, user_id, workspace_id
                   FROM webhook_configs WHERE 1=1"""
        params: List[Any] = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        if active_only:
            query += " AND active = 1"

        query += " ORDER BY created_at DESC"

        cursor = conn.execute(query, params)
        return [WebhookConfig.from_row(row) for row in cursor.fetchall()]

    def delete(self, webhook_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM webhook_configs WHERE id = ?", (webhook_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted webhook {webhook_id}")
        return deleted

    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[WebhookConfig]:
        webhook = self.get(webhook_id)
        if not webhook:
            return None

        updates: List[str] = []
        params: List[Any] = []

        if url is not None:
            updates.append("url = ?")
            params.append(url)
            webhook.url = url
        if events is not None:
            updates.append("events_json = ?")
            params.append(json.dumps(events))
            webhook.events = events
        if active is not None:
            updates.append("active = ?")
            params.append(int(active))
            webhook.active = active
        if name is not None:
            updates.append("name = ?")
            params.append(name)
            webhook.name = name
        if description is not None:
            updates.append("description = ?")
            params.append(description)
            webhook.description = description

        if updates:
            updates.append("updated_at = ?")
            params.append(time.time())
            params.append(webhook_id)

            conn = self._get_conn()
            conn.execute(
                f"UPDATE webhook_configs SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            webhook.updated_at = time.time()

        return webhook

    def record_delivery(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        conn = self._get_conn()
        if success:
            conn.execute(
                """UPDATE webhook_configs SET
                   last_delivery_at = ?, last_delivery_status = ?,
                   delivery_count = delivery_count + 1
                   WHERE id = ?""",
                (time.time(), status_code, webhook_id),
            )
        else:
            conn.execute(
                """UPDATE webhook_configs SET
                   last_delivery_at = ?, last_delivery_status = ?,
                   delivery_count = delivery_count + 1, failure_count = failure_count + 1
                   WHERE id = ?""",
                (time.time(), status_code, webhook_id),
            )
        conn.commit()

    def get_for_event(self, event_type: str) -> List[WebhookConfig]:
        conn = self._get_conn()
        cursor = conn.execute(
            """SELECT id, url, events_json, secret, active, created_at, updated_at,
                      name, description, last_delivery_at, last_delivery_status,
                      delivery_count, failure_count, user_id, workspace_id
               FROM webhook_configs WHERE active = 1"""
        )
        webhooks = [WebhookConfig.from_row(row) for row in cursor.fetchall()]
        return [w for w in webhooks if w.matches_event(event_type)]

    def close(self) -> None:
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


class RedisWebhookConfigStore(WebhookConfigStoreBackend):
    """
    Redis-backed webhook config store with SQLite fallback.

    Uses Redis for fast distributed access, with SQLite as durable storage.
    This enables multi-instance deployments while ensuring persistence.
    """

    REDIS_PREFIX = "aragora:webhook_configs"
    REDIS_TTL = 86400  # 24 hours

    def __init__(self, db_path: Path | str, redis_url: Optional[str] = None):
        self._sqlite = SQLiteWebhookConfigStore(db_path)
        self._redis: Optional[Any] = None
        self._redis_url = redis_url or os.environ.get(
            "ARAGORA_REDIS_URL", "redis://localhost:6379"
        )
        self._redis_checked = False
        logger.info("RedisWebhookConfigStore initialized with SQLite fallback")

    def _get_redis(self) -> Optional[Any]:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
            self._redis.ping()
            self._redis_checked = True
            logger.info("Redis connected for webhook config store")
        except Exception as e:
            logger.debug(f"Redis not available, using SQLite only: {e}")
            self._redis = None
            self._redis_checked = True

        return self._redis

    def _redis_key(self, webhook_id: str) -> str:
        return f"{self.REDIS_PREFIX}:{webhook_id}"

    def register(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> WebhookConfig:
        # Always save to SQLite (durable)
        webhook = self._sqlite.register(
            url=url,
            events=events,
            name=name,
            description=description,
            user_id=user_id,
            workspace_id=workspace_id,
        )

        # Update Redis cache
        redis = self._get_redis()
        if redis:
            try:
                redis.setex(
                    self._redis_key(webhook.id), self.REDIS_TTL, webhook.to_json()
                )
            except Exception as e:
                logger.debug(f"Redis cache update failed: {e}")

        return webhook

    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        redis = self._get_redis()

        # Try Redis first
        if redis is not None:
            try:
                data = redis.get(self._redis_key(webhook_id))
                if data:
                    return WebhookConfig.from_json(data)
            except Exception as e:
                logger.debug(f"Redis get failed, falling back to SQLite: {e}")

        # Fall back to SQLite
        webhook = self._sqlite.get(webhook_id)

        # Populate Redis cache if found
        if webhook and redis:
            try:
                redis.setex(
                    self._redis_key(webhook_id), self.REDIS_TTL, webhook.to_json()
                )
            except (ConnectionError, TimeoutError) as e:
                logger.debug(f"Redis cache population failed (connection issue): {e}")
            except Exception as e:
                logger.debug(f"Redis cache population failed: {e}")

        return webhook

    def list(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        # Always use SQLite for list operations (authoritative)
        return self._sqlite.list(
            user_id=user_id, workspace_id=workspace_id, active_only=active_only
        )

    def delete(self, webhook_id: str) -> bool:
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._redis_key(webhook_id))
            except (ConnectionError, TimeoutError) as e:
                logger.debug(f"Redis cache delete failed (connection issue): {e}")
            except Exception as e:
                logger.debug(f"Redis cache delete failed: {e}")

        return self._sqlite.delete(webhook_id)

    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[WebhookConfig]:
        webhook = self._sqlite.update(
            webhook_id=webhook_id,
            url=url,
            events=events,
            active=active,
            name=name,
            description=description,
        )

        # Update Redis cache
        if webhook:
            redis = self._get_redis()
            if redis:
                try:
                    redis.setex(
                        self._redis_key(webhook_id), self.REDIS_TTL, webhook.to_json()
                    )
                except Exception as e:
                    logger.debug(f"Redis cache update failed: {e}")

        return webhook

    def record_delivery(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        self._sqlite.record_delivery(webhook_id, status_code, success)

        # Invalidate Redis cache (next get will refresh)
        redis = self._get_redis()
        if redis:
            try:
                redis.delete(self._redis_key(webhook_id))
            except Exception as e:
                logger.debug(f"Redis cache invalidation failed: {e}")

    def get_for_event(self, event_type: str) -> List[WebhookConfig]:
        return self._sqlite.get_for_event(event_type)

    def close(self) -> None:
        self._sqlite.close()
        if self._redis:
            self._redis.close()


class PostgresWebhookConfigStore(WebhookConfigStoreBackend):
    """
    PostgreSQL-backed webhook config store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "webhook_configs"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS webhook_configs (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            events_json JSONB NOT NULL,
            secret TEXT NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            name TEXT,
            description TEXT,
            last_delivery_at TIMESTAMPTZ,
            last_delivery_status INTEGER,
            delivery_count INTEGER DEFAULT 0,
            failure_count INTEGER DEFAULT 0,
            user_id TEXT,
            workspace_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_webhook_configs_user ON webhook_configs(user_id);
        CREATE INDEX IF NOT EXISTS idx_webhook_configs_workspace ON webhook_configs(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_webhook_configs_active ON webhook_configs(active);
    """

    def __init__(self, pool: "Pool"):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresWebhookConfigStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    def register(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> WebhookConfig:
        """Register a new webhook (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.register_async(url, events, name, description, user_id, workspace_id)
        )

    async def register_async(
        self,
        url: str,
        events: List[str],
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> WebhookConfig:
        """Register a new webhook asynchronously."""
        webhook_id = str(uuid.uuid4())
        secret = secrets.token_urlsafe(32)
        now = time.time()

        webhook = WebhookConfig(
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

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO webhook_configs
                   (id, url, events_json, secret, active, created_at, updated_at,
                    name, description, last_delivery_at, last_delivery_status,
                    delivery_count, failure_count, user_id, workspace_id)
                   VALUES ($1, $2, $3, $4, $5, to_timestamp($6), to_timestamp($7),
                           $8, $9, $10, $11, $12, $13, $14, $15)""",
                webhook.id,
                webhook.url,
                json.dumps(webhook.events),
                webhook.secret,
                webhook.active,
                webhook.created_at,
                webhook.updated_at,
                webhook.name,
                webhook.description,
                None,  # last_delivery_at
                None,  # last_delivery_status
                0,  # delivery_count
                0,  # failure_count
                webhook.user_id,
                webhook.workspace_id,
            )

        logger.info(f"Registered webhook {webhook_id} for events: {events}")
        return webhook

    def get(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.get_async(webhook_id))

    async def get_async(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, url, events_json, secret, active,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          name, description,
                          EXTRACT(EPOCH FROM last_delivery_at) as last_delivery_at,
                          last_delivery_status, delivery_count, failure_count,
                          user_id, workspace_id
                   FROM webhook_configs WHERE id = $1""",
                webhook_id,
            )
            if row:
                return self._row_to_config(row)
            return None

    def _row_to_config(self, row: Any) -> WebhookConfig:
        """Convert database row to WebhookConfig."""
        return WebhookConfig(
            id=row["id"],
            url=row["url"],
            events=json.loads(row["events_json"]) if row["events_json"] else [],
            secret=row["secret"],
            active=bool(row["active"]),
            created_at=row["created_at"] or time.time(),
            updated_at=row["updated_at"] or time.time(),
            name=row["name"],
            description=row["description"],
            last_delivery_at=row["last_delivery_at"],
            last_delivery_status=row["last_delivery_status"],
            delivery_count=row["delivery_count"] or 0,
            failure_count=row["failure_count"] or 0,
            user_id=row["user_id"],
            workspace_id=row["workspace_id"],
        )

    def list(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        """List webhooks (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_async(user_id, workspace_id, active_only)
        )

    async def list_async(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        active_only: bool = False,
    ) -> List[WebhookConfig]:
        """List webhooks asynchronously."""
        query = """SELECT id, url, events_json, secret, active,
                          EXTRACT(EPOCH FROM created_at) as created_at,
                          EXTRACT(EPOCH FROM updated_at) as updated_at,
                          name, description,
                          EXTRACT(EPOCH FROM last_delivery_at) as last_delivery_at,
                          last_delivery_status, delivery_count, failure_count,
                          user_id, workspace_id
                   FROM webhook_configs WHERE 1=1"""
        params: List[Any] = []
        param_idx = 1

        if user_id:
            query += f" AND user_id = ${param_idx}"
            params.append(user_id)
            param_idx += 1
        if workspace_id:
            query += f" AND workspace_id = ${param_idx}"
            params.append(workspace_id)
            param_idx += 1
        if active_only:
            query += " AND active = TRUE"

        query += " ORDER BY created_at DESC"

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_config(row) for row in rows]

    def delete(self, webhook_id: str) -> bool:
        """Delete webhook (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.delete_async(webhook_id))

    async def delete_async(self, webhook_id: str) -> bool:
        """Delete webhook asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM webhook_configs WHERE id = $1", webhook_id
            )
            deleted = result != "DELETE 0"
            if deleted:
                logger.info(f"Deleted webhook {webhook_id}")
            return deleted

    def update(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[WebhookConfig]:
        """Update webhook (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_async(webhook_id, url, events, active, name, description)
        )

    async def update_async(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        active: Optional[bool] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[WebhookConfig]:
        """Update webhook asynchronously."""
        webhook = await self.get_async(webhook_id)
        if not webhook:
            return None

        updates: List[str] = []
        params: List[Any] = []
        param_idx = 1

        if url is not None:
            updates.append(f"url = ${param_idx}")
            params.append(url)
            param_idx += 1
            webhook.url = url
        if events is not None:
            updates.append(f"events_json = ${param_idx}")
            params.append(json.dumps(events))
            param_idx += 1
            webhook.events = events
        if active is not None:
            updates.append(f"active = ${param_idx}")
            params.append(active)
            param_idx += 1
            webhook.active = active
        if name is not None:
            updates.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1
            webhook.name = name
        if description is not None:
            updates.append(f"description = ${param_idx}")
            params.append(description)
            param_idx += 1
            webhook.description = description

        if updates:
            updates.append(f"updated_at = to_timestamp(${param_idx})")
            params.append(time.time())
            param_idx += 1
            params.append(webhook_id)

            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"UPDATE webhook_configs SET {', '.join(updates)} WHERE id = ${param_idx}",
                    *params,
                )
            webhook.updated_at = time.time()

        return webhook

    def record_delivery(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        """Record delivery (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(
            self.record_delivery_async(webhook_id, status_code, success)
        )

    async def record_delivery_async(
        self,
        webhook_id: str,
        status_code: int,
        success: bool = True,
    ) -> None:
        """Record delivery asynchronously."""
        async with self._pool.acquire() as conn:
            if success:
                await conn.execute(
                    """UPDATE webhook_configs SET
                       last_delivery_at = NOW(), last_delivery_status = $1,
                       delivery_count = delivery_count + 1
                       WHERE id = $2""",
                    status_code,
                    webhook_id,
                )
            else:
                await conn.execute(
                    """UPDATE webhook_configs SET
                       last_delivery_at = NOW(), last_delivery_status = $1,
                       delivery_count = delivery_count + 1, failure_count = failure_count + 1
                       WHERE id = $2""",
                    status_code,
                    webhook_id,
                )

    def get_for_event(self, event_type: str) -> List[WebhookConfig]:
        """Get webhooks for event (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_for_event_async(event_type)
        )

    async def get_for_event_async(self, event_type: str) -> List[WebhookConfig]:
        """Get webhooks for event asynchronously."""
        webhooks = await self.list_async(active_only=True)
        return [w for w in webhooks if w.matches_event(event_type)]

    def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# =============================================================================
# Global Store Factory
# =============================================================================

_webhook_config_store: Optional[WebhookConfigStoreBackend] = None


def get_webhook_config_store() -> WebhookConfigStoreBackend:
    """
    Get or create the webhook config store.

    Uses environment variables to configure:
    - ARAGORA_WEBHOOK_CONFIG_STORE_BACKEND: "memory", "sqlite", or "redis" (default: sqlite)
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_REDIS_URL: Redis connection URL (for redis backend)

    Returns:
        Configured WebhookConfigStoreBackend instance
    """
    global _webhook_config_store
    if _webhook_config_store is not None:
        return _webhook_config_store

    backend_type = os.environ.get("ARAGORA_WEBHOOK_CONFIG_STORE_BACKEND", "sqlite").lower()

    # Get data directory
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
        data_dir = Path(env_dir or ".nomic")

    db_path = data_dir / "webhook_configs.db"

    if backend_type == "memory":
        logger.info("Using in-memory webhook config store (not persistent)")
        _webhook_config_store = InMemoryWebhookConfigStore()
    elif backend_type == "redis":
        logger.info("Using Redis webhook config store with SQLite fallback")
        _webhook_config_store = RedisWebhookConfigStore(db_path)
    else:  # Default: sqlite
        logger.info(f"Using SQLite webhook config store: {db_path}")
        _webhook_config_store = SQLiteWebhookConfigStore(db_path)

    return _webhook_config_store


def set_webhook_config_store(store: WebhookConfigStoreBackend) -> None:
    """
    Set custom webhook config store.

    Useful for testing or custom deployments.
    """
    global _webhook_config_store
    _webhook_config_store = store
    logger.debug(f"Webhook config store backend set: {type(store).__name__}")


def reset_webhook_config_store() -> None:
    """Reset the global webhook config store (for testing)."""
    global _webhook_config_store
    _webhook_config_store = None


__all__ = [
    "WebhookConfig",
    "WebhookConfigStoreBackend",
    "InMemoryWebhookConfigStore",
    "SQLiteWebhookConfigStore",
    "RedisWebhookConfigStore",
    "get_webhook_config_store",
    "set_webhook_config_store",
    "reset_webhook_config_store",
    "WEBHOOK_EVENTS",
]
