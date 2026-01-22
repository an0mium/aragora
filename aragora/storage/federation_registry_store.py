"""
Federation Registry Storage Backends.

Persistent storage for federated region configurations used by Knowledge Mound.

Backends:
- InMemoryFederationRegistryStore: For testing
- SQLiteFederationRegistryStore: For single-instance deployments
- RedisFederationRegistryStore: For multi-instance (with SQLite fallback)
- PostgresFederationRegistryStore: For multi-instance PostgreSQL deployments

Usage:
    from aragora.storage.federation_registry_store import (
        get_federation_registry_store,
        set_federation_registry_store,
    )

    store = get_federation_registry_store()
    await store.save(federated_region)
    region = await store.get("us-west-2")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)

# Global singleton
_federation_registry_store: Optional["FederationRegistryStoreBackend"] = None
_store_lock = threading.RLock()


@dataclass
class FederatedRegionConfig:
    """
    Configuration for a federated region.

    Stores region connection info, sync settings, and status.
    """

    region_id: str
    endpoint_url: str
    api_key: str  # Encrypted in production
    mode: str = "bidirectional"  # push, pull, bidirectional, none
    sync_scope: str = "summary"  # full, metadata, summary
    enabled: bool = True
    workspace_id: Optional[str] = None

    # Sync status
    last_sync_at: Optional[str] = None
    last_sync_error: Optional[str] = None
    last_push_at: Optional[str] = None
    last_pull_at: Optional[str] = None

    # Metrics
    total_pushes: int = 0
    total_pulls: int = 0
    total_nodes_synced: int = 0
    total_sync_errors: int = 0

    # Timestamps
    created_at: str = ""
    updated_at: str = ""

    # Additional configuration
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default timestamps if not provided."""
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "region_id": self.region_id,
            "endpoint_url": self.endpoint_url,
            "api_key": self.api_key,
            "mode": self.mode,
            "sync_scope": self.sync_scope,
            "enabled": self.enabled,
            "workspace_id": self.workspace_id,
            "last_sync_at": self.last_sync_at,
            "last_sync_error": self.last_sync_error,
            "last_push_at": self.last_push_at,
            "last_pull_at": self.last_pull_at,
            "total_pushes": self.total_pushes,
            "total_pulls": self.total_pulls,
            "total_nodes_synced": self.total_nodes_synced,
            "total_sync_errors": self.total_sync_errors,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FederatedRegionConfig":
        """Create from dictionary."""
        return cls(
            region_id=data.get("region_id", ""),
            endpoint_url=data.get("endpoint_url", ""),
            api_key=data.get("api_key", ""),
            mode=data.get("mode", "bidirectional"),
            sync_scope=data.get("sync_scope", "summary"),
            enabled=data.get("enabled", True),
            workspace_id=data.get("workspace_id"),
            last_sync_at=data.get("last_sync_at"),
            last_sync_error=data.get("last_sync_error"),
            last_push_at=data.get("last_push_at"),
            last_pull_at=data.get("last_pull_at"),
            total_pushes=data.get("total_pushes", 0),
            total_pulls=data.get("total_pulls", 0),
            total_nodes_synced=data.get("total_nodes_synced", 0),
            total_sync_errors=data.get("total_sync_errors", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "FederatedRegionConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))


class FederationRegistryStoreBackend(ABC):
    """Abstract base class for federation registry storage backends."""

    @abstractmethod
    async def get(
        self, region_id: str, workspace_id: Optional[str] = None
    ) -> Optional[FederatedRegionConfig]:
        """Get a federated region by ID."""
        pass

    @abstractmethod
    async def save(self, region: FederatedRegionConfig) -> None:
        """Save a federated region configuration."""
        pass

    @abstractmethod
    async def delete(self, region_id: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a federated region."""
        pass

    @abstractmethod
    async def list_all(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List all federated regions."""
        pass

    @abstractmethod
    async def list_enabled(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List enabled federated regions."""
        pass

    @abstractmethod
    async def update_sync_status(
        self,
        region_id: str,
        direction: str,  # "push" or "pull"
        nodes_synced: int = 0,
        error: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Update sync status after a sync operation."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources."""
        pass


class InMemoryFederationRegistryStore(FederationRegistryStoreBackend):
    """
    In-memory federation registry store for testing.

    Data is lost on restart.
    """

    def __init__(self) -> None:
        """Initialize in-memory store."""
        self._data: Dict[str, FederatedRegionConfig] = {}
        self._lock = threading.RLock()

    def _make_key(self, region_id: str, workspace_id: Optional[str]) -> str:
        """Create a composite key for workspace-scoped regions."""
        if workspace_id:
            return f"{workspace_id}:{region_id}"
        return region_id

    async def get(
        self, region_id: str, workspace_id: Optional[str] = None
    ) -> Optional[FederatedRegionConfig]:
        """Get a federated region by ID."""
        key = self._make_key(region_id, workspace_id)
        with self._lock:
            return self._data.get(key)

    async def save(self, region: FederatedRegionConfig) -> None:
        """Save a federated region configuration."""
        region.updated_at = datetime.now(timezone.utc).isoformat()
        key = self._make_key(region.region_id, region.workspace_id)
        with self._lock:
            self._data[key] = region

    async def delete(self, region_id: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a federated region."""
        key = self._make_key(region_id, workspace_id)
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def list_all(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List all federated regions."""
        with self._lock:
            if workspace_id:
                return [r for r in self._data.values() if r.workspace_id == workspace_id]
            return list(self._data.values())

    async def list_enabled(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List enabled federated regions."""
        all_regions = await self.list_all(workspace_id)
        return [r for r in all_regions if r.enabled]

    async def update_sync_status(
        self,
        region_id: str,
        direction: str,
        nodes_synced: int = 0,
        error: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Update sync status after a sync operation."""
        region = await self.get(region_id, workspace_id)
        if region:
            now = datetime.now(timezone.utc).isoformat()
            region.last_sync_at = now
            region.last_sync_error = error
            region.updated_at = now

            if direction == "push":
                region.last_push_at = now
                region.total_pushes += 1
            else:
                region.last_pull_at = now
                region.total_pulls += 1

            if error:
                region.total_sync_errors += 1
            else:
                region.total_nodes_synced += nodes_synced

            await self.save(region)

    async def close(self) -> None:
        """No-op for in-memory store."""
        pass


class SQLiteFederationRegistryStore(FederationRegistryStoreBackend):
    """
    SQLite-backed federation registry store.

    Suitable for single-instance deployments.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database. Defaults to
                     $ARAGORA_DATA_DIR/federation_registry.db
        """
        if db_path is None:
            data_dir = os.getenv("ARAGORA_DATA_DIR", "/tmp/aragora")
            db_path = Path(data_dir) / "federation_registry.db"

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS federated_regions (
                        region_id TEXT NOT NULL,
                        workspace_id TEXT,
                        endpoint_url TEXT NOT NULL,
                        api_key TEXT NOT NULL,
                        mode TEXT DEFAULT 'bidirectional',
                        sync_scope TEXT DEFAULT 'summary',
                        enabled BOOLEAN DEFAULT TRUE,
                        last_sync_at TEXT,
                        last_sync_error TEXT,
                        last_push_at TEXT,
                        last_pull_at TEXT,
                        total_pushes INTEGER DEFAULT 0,
                        total_pulls INTEGER DEFAULT 0,
                        total_nodes_synced INTEGER DEFAULT 0,
                        total_sync_errors INTEGER DEFAULT 0,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        data_json TEXT NOT NULL,
                        PRIMARY KEY (region_id, workspace_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_federation_workspace
                    ON federated_regions(workspace_id)
                    """
                )
                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_federation_enabled
                    ON federated_regions(enabled)
                    """
                )
                conn.commit()
            finally:
                conn.close()

    async def get(
        self, region_id: str, workspace_id: Optional[str] = None
    ) -> Optional[FederatedRegionConfig]:
        """Get a federated region by ID."""
        ws_id = workspace_id or ""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM federated_regions WHERE region_id = ? AND workspace_id = ?",
                    (region_id, ws_id),
                )
                row = cursor.fetchone()
                if row:
                    return FederatedRegionConfig.from_json(row[0])
                return None
            finally:
                conn.close()

    async def save(self, region: FederatedRegionConfig) -> None:
        """Save a federated region configuration."""
        now = time.time()
        region.updated_at = datetime.now(timezone.utc).isoformat()
        data_json = region.to_json()
        ws_id = region.workspace_id or ""

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO federated_regions
                    (region_id, workspace_id, endpoint_url, api_key, mode, sync_scope,
                     enabled, last_sync_at, last_sync_error, last_push_at, last_pull_at,
                     total_pushes, total_pulls, total_nodes_synced, total_sync_errors,
                     created_at, updated_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        region.region_id,
                        ws_id,
                        region.endpoint_url,
                        region.api_key,
                        region.mode,
                        region.sync_scope,
                        region.enabled,
                        region.last_sync_at,
                        region.last_sync_error,
                        region.last_push_at,
                        region.last_pull_at,
                        region.total_pushes,
                        region.total_pulls,
                        region.total_nodes_synced,
                        region.total_sync_errors,
                        now,
                        now,
                        data_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def delete(self, region_id: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a federated region."""
        ws_id = workspace_id or ""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "DELETE FROM federated_regions WHERE region_id = ? AND workspace_id = ?",
                    (region_id, ws_id),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def list_all(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List all federated regions."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                if workspace_id:
                    cursor.execute(
                        "SELECT data_json FROM federated_regions WHERE workspace_id = ?",
                        (workspace_id,),
                    )
                else:
                    cursor.execute("SELECT data_json FROM federated_regions")
                return [FederatedRegionConfig.from_json(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def list_enabled(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List enabled federated regions."""
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                if workspace_id:
                    cursor.execute(
                        "SELECT data_json FROM federated_regions WHERE workspace_id = ? AND enabled = TRUE",
                        (workspace_id,),
                    )
                else:
                    cursor.execute("SELECT data_json FROM federated_regions WHERE enabled = TRUE")
                return [FederatedRegionConfig.from_json(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def update_sync_status(
        self,
        region_id: str,
        direction: str,
        nodes_synced: int = 0,
        error: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Update sync status after a sync operation."""
        region = await self.get(region_id, workspace_id)
        if region:
            now = datetime.now(timezone.utc).isoformat()
            region.last_sync_at = now
            region.last_sync_error = error

            if direction == "push":
                region.last_push_at = now
                region.total_pushes += 1
            else:
                region.last_pull_at = now
                region.total_pulls += 1

            if error:
                region.total_sync_errors += 1
            else:
                region.total_nodes_synced += nodes_synced

            await self.save(region)

    async def close(self) -> None:
        """No-op for SQLite (connections are per-operation)."""
        pass


class RedisFederationRegistryStore(FederationRegistryStoreBackend):
    """
    Redis-backed federation registry store with SQLite fallback.

    For multi-instance deployments with optional horizontal scaling.
    Falls back to SQLite if Redis is unavailable.
    """

    REDIS_PREFIX = "aragora:federation:"

    def __init__(
        self,
        fallback_db_path: Optional[Path] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        """
        Initialize Redis store with SQLite fallback.

        Args:
            fallback_db_path: Path for SQLite fallback database
            redis_url: Redis connection URL (defaults to ARAGORA_REDIS_URL env var)
        """
        self._redis_url = redis_url or os.getenv("ARAGORA_REDIS_URL", "")
        self._redis_client: Any = None
        self._fallback = SQLiteFederationRegistryStore(fallback_db_path)
        self._using_fallback = False
        self._lock = threading.RLock()

        self._connect_redis()

    def _connect_redis(self) -> None:
        """Attempt to connect to Redis."""
        if not self._redis_url:
            logger.info("No Redis URL configured, using SQLite fallback")
            self._using_fallback = True
            return

        try:
            import redis

            self._redis_client = redis.from_url(self._redis_url)
            self._redis_client.ping()
            logger.info("Connected to Redis for federation registry storage")
            self._using_fallback = False
        except Exception as e:
            logger.warning(f"Redis connection failed, using SQLite fallback: {e}")
            self._using_fallback = True
            self._redis_client = None

    def _make_key(self, region_id: str, workspace_id: Optional[str]) -> str:
        """Create Redis key for a region."""
        ws_id = workspace_id or "_global"
        return f"{self.REDIS_PREFIX}{ws_id}:{region_id}"

    async def get(
        self, region_id: str, workspace_id: Optional[str] = None
    ) -> Optional[FederatedRegionConfig]:
        """Get a federated region by ID."""
        if self._using_fallback:
            return await self._fallback.get(region_id, workspace_id)

        try:
            key = self._make_key(region_id, workspace_id)
            data = self._redis_client.get(key)
            if data:
                return FederatedRegionConfig.from_json(data)
            return None
        except Exception as e:
            logger.warning(f"Redis get failed, using fallback: {e}")
            return await self._fallback.get(region_id, workspace_id)

    async def save(self, region: FederatedRegionConfig) -> None:
        """Save a federated region configuration."""
        region.updated_at = datetime.now(timezone.utc).isoformat()

        # Always save to SQLite fallback for durability
        await self._fallback.save(region)

        if self._using_fallback:
            return

        try:
            key = self._make_key(region.region_id, region.workspace_id)
            self._redis_client.set(key, region.to_json())
        except Exception as e:
            logger.warning(f"Redis save failed (SQLite fallback used): {e}")

    async def delete(self, region_id: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a federated region."""
        result = await self._fallback.delete(region_id, workspace_id)

        if self._using_fallback:
            return result

        try:
            key = self._make_key(region_id, workspace_id)
            self._redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")

        return result

    async def list_all(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List all federated regions."""
        # Use SQLite for listing (more efficient for scanning)
        return await self._fallback.list_all(workspace_id)

    async def list_enabled(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List enabled federated regions."""
        return await self._fallback.list_enabled(workspace_id)

    async def update_sync_status(
        self,
        region_id: str,
        direction: str,
        nodes_synced: int = 0,
        error: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Update sync status after a sync operation."""
        # Get current region
        region = await self.get(region_id, workspace_id)
        if region:
            now = datetime.now(timezone.utc).isoformat()
            region.last_sync_at = now
            region.last_sync_error = error

            if direction == "push":
                region.last_push_at = now
                region.total_pushes += 1
            else:
                region.last_pull_at = now
                region.total_pulls += 1

            if error:
                region.total_sync_errors += 1
            else:
                region.total_nodes_synced += nodes_synced

            await self.save(region)

    async def close(self) -> None:
        """Close connections."""
        await self._fallback.close()
        if self._redis_client:
            try:
                self._redis_client.close()
            except (ConnectionError, OSError) as e:
                logger.debug(f"Redis close failed (connection already closed): {e}")
            except Exception as e:
                logger.debug(f"Redis close failed: {e}")


class PostgresFederationRegistryStore(FederationRegistryStoreBackend):
    """
    PostgreSQL-backed federation registry store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "federation_registry"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS federated_regions (
            region_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT '',
            endpoint_url TEXT NOT NULL,
            api_key TEXT NOT NULL,
            mode TEXT DEFAULT 'bidirectional',
            sync_scope TEXT DEFAULT 'summary',
            enabled BOOLEAN DEFAULT TRUE,
            last_sync_at TEXT,
            last_sync_error TEXT,
            last_push_at TEXT,
            last_pull_at TEXT,
            total_pushes INTEGER DEFAULT 0,
            total_pulls INTEGER DEFAULT 0,
            total_nodes_synced INTEGER DEFAULT 0,
            total_sync_errors INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            data_json JSONB NOT NULL,
            PRIMARY KEY (region_id, workspace_id)
        );

        CREATE INDEX IF NOT EXISTS idx_federation_workspace
        ON federated_regions(workspace_id);

        CREATE INDEX IF NOT EXISTS idx_federation_enabled
        ON federated_regions(enabled);
    """

    def __init__(self, pool: "Pool"):
        """Initialize PostgreSQL federation registry store.

        Args:
            pool: asyncpg connection pool
        """
        self._pool = pool
        self._initialized = False
        logger.info("PostgresFederationRegistryStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    async def get(
        self, region_id: str, workspace_id: Optional[str] = None
    ) -> Optional[FederatedRegionConfig]:
        """Get a federated region by ID."""
        ws_id = workspace_id or ""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM federated_regions WHERE region_id = $1 AND workspace_id = $2",
                region_id,
                ws_id,
            )
            if row:
                data = row["data_json"]
                if isinstance(data, str):
                    return FederatedRegionConfig.from_json(data)
                else:
                    return FederatedRegionConfig.from_dict(data)
            return None

    def get_sync(
        self, region_id: str, workspace_id: Optional[str] = None
    ) -> Optional[FederatedRegionConfig]:
        """Get a federated region by ID (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.get(region_id, workspace_id))

    async def save(self, region: FederatedRegionConfig) -> None:
        """Save a federated region configuration."""
        now = time.time()
        region.updated_at = datetime.now(timezone.utc).isoformat()
        data_json = region.to_json()
        ws_id = region.workspace_id or ""

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO federated_regions
                (region_id, workspace_id, endpoint_url, api_key, mode, sync_scope,
                 enabled, last_sync_at, last_sync_error, last_push_at, last_pull_at,
                 total_pushes, total_pulls, total_nodes_synced, total_sync_errors,
                 created_at, updated_at, data_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, to_timestamp($16), to_timestamp($17), $18)
                ON CONFLICT (region_id, workspace_id) DO UPDATE SET
                    endpoint_url = EXCLUDED.endpoint_url,
                    api_key = EXCLUDED.api_key,
                    mode = EXCLUDED.mode,
                    sync_scope = EXCLUDED.sync_scope,
                    enabled = EXCLUDED.enabled,
                    last_sync_at = EXCLUDED.last_sync_at,
                    last_sync_error = EXCLUDED.last_sync_error,
                    last_push_at = EXCLUDED.last_push_at,
                    last_pull_at = EXCLUDED.last_pull_at,
                    total_pushes = EXCLUDED.total_pushes,
                    total_pulls = EXCLUDED.total_pulls,
                    total_nodes_synced = EXCLUDED.total_nodes_synced,
                    total_sync_errors = EXCLUDED.total_sync_errors,
                    updated_at = EXCLUDED.updated_at,
                    data_json = EXCLUDED.data_json
                """,
                region.region_id,
                ws_id,
                region.endpoint_url,
                region.api_key,
                region.mode,
                region.sync_scope,
                region.enabled,
                region.last_sync_at,
                region.last_sync_error,
                region.last_push_at,
                region.last_pull_at,
                region.total_pushes,
                region.total_pulls,
                region.total_nodes_synced,
                region.total_sync_errors,
                now,
                now,
                data_json,
            )

    def save_sync(self, region: FederatedRegionConfig) -> None:
        """Save a federated region configuration (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(self.save(region))

    async def delete(self, region_id: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a federated region."""
        ws_id = workspace_id or ""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM federated_regions WHERE region_id = $1 AND workspace_id = $2",
                region_id,
                ws_id,
            )
            return result != "DELETE 0"

    def delete_sync(self, region_id: str, workspace_id: Optional[str] = None) -> bool:
        """Delete a federated region (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.delete(region_id, workspace_id))

    async def list_all(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List all federated regions."""
        async with self._pool.acquire() as conn:
            if workspace_id:
                rows = await conn.fetch(
                    "SELECT data_json FROM federated_regions WHERE workspace_id = $1",
                    workspace_id,
                )
            else:
                rows = await conn.fetch("SELECT data_json FROM federated_regions")

            results = []
            for row in rows:
                data = row["data_json"]
                if isinstance(data, str):
                    results.append(FederatedRegionConfig.from_json(data))
                else:
                    results.append(FederatedRegionConfig.from_dict(data))
            return results

    def list_all_sync(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List all federated regions (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.list_all(workspace_id))

    async def list_enabled(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List enabled federated regions."""
        async with self._pool.acquire() as conn:
            if workspace_id:
                rows = await conn.fetch(
                    "SELECT data_json FROM federated_regions WHERE workspace_id = $1 AND enabled = TRUE",
                    workspace_id,
                )
            else:
                rows = await conn.fetch(
                    "SELECT data_json FROM federated_regions WHERE enabled = TRUE"
                )

            results = []
            for row in rows:
                data = row["data_json"]
                if isinstance(data, str):
                    results.append(FederatedRegionConfig.from_json(data))
                else:
                    results.append(FederatedRegionConfig.from_dict(data))
            return results

    def list_enabled_sync(self, workspace_id: Optional[str] = None) -> List[FederatedRegionConfig]:
        """List enabled federated regions (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.list_enabled(workspace_id))

    async def update_sync_status(
        self,
        region_id: str,
        direction: str,
        nodes_synced: int = 0,
        error: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Update sync status after a sync operation.

        Uses atomic UPDATE to avoid read-modify-write race conditions.
        """
        ws_id = workspace_id or ""
        now = datetime.now(timezone.utc).isoformat()

        # Build atomic update based on direction and error status
        if direction == "push":
            if error:
                sql = """
                    UPDATE federated_regions SET
                        last_sync_at = $1, last_sync_error = $2, last_push_at = $1,
                        total_pushes = total_pushes + 1, total_sync_errors = total_sync_errors + 1,
                        updated_at = NOW()
                    WHERE region_id = $3 AND workspace_id = $4
                """
                params = (now, error, region_id, ws_id)
            else:
                sql = """
                    UPDATE federated_regions SET
                        last_sync_at = $1, last_sync_error = NULL, last_push_at = $1,
                        total_pushes = total_pushes + 1, total_nodes_synced = total_nodes_synced + $2,
                        updated_at = NOW()
                    WHERE region_id = $3 AND workspace_id = $4
                """
                params = (now, nodes_synced, region_id, ws_id)  # type: ignore[assignment]
        else:  # pull
            if error:
                sql = """
                    UPDATE federated_regions SET
                        last_sync_at = $1, last_sync_error = $2, last_pull_at = $1,
                        total_pulls = total_pulls + 1, total_sync_errors = total_sync_errors + 1,
                        updated_at = NOW()
                    WHERE region_id = $3 AND workspace_id = $4
                """
                params = (now, error, region_id, ws_id)
            else:
                sql = """
                    UPDATE federated_regions SET
                        last_sync_at = $1, last_sync_error = NULL, last_pull_at = $1,
                        total_pulls = total_pulls + 1, total_nodes_synced = total_nodes_synced + $2,
                        updated_at = NOW()
                    WHERE region_id = $3 AND workspace_id = $4
                """
                params = (now, nodes_synced, region_id, ws_id)  # type: ignore[assignment]

        async with self._pool.acquire() as conn:
            await conn.execute(sql, *params)

        # Update data_json field to keep it in sync
        region = await self.get(region_id, workspace_id)
        if region:
            await self.save(region)

    def update_sync_status_sync(
        self,
        region_id: str,
        direction: str,
        nodes_synced: int = 0,
        error: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> None:
        """Update sync status (sync wrapper for async)."""
        asyncio.get_event_loop().run_until_complete(
            self.update_sync_status(region_id, direction, nodes_synced, error, workspace_id)
        )

    async def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


def get_federation_registry_store() -> FederationRegistryStoreBackend:
    """
    Get the global federation registry store instance.

    Backend is selected based on environment variables:
    - ARAGORA_FEDERATION_STORE_BACKEND: Store-specific override
    - ARAGORA_DB_BACKEND: Global database backend fallback

    Values:
    - "memory": InMemoryFederationRegistryStore (for testing)
    - "sqlite": SQLiteFederationRegistryStore (default, single-instance)
    - "redis": RedisFederationRegistryStore (multi-instance)
    - "postgres" or "postgresql": PostgresFederationRegistryStore (multi-instance)
    """
    global _federation_registry_store

    with _store_lock:
        if _federation_registry_store is not None:
            return _federation_registry_store

        # Check store-specific backend first, then global database backend
        backend = os.getenv("ARAGORA_FEDERATION_STORE_BACKEND")
        if not backend:
            backend = os.getenv("ARAGORA_DB_BACKEND", "sqlite")
        backend = backend.lower()

        if backend == "memory":
            _federation_registry_store = InMemoryFederationRegistryStore()
            logger.info("Using in-memory federation registry store")
        elif backend == "redis":
            _federation_registry_store = RedisFederationRegistryStore()
            logger.info("Using Redis federation registry store")
        elif backend == "postgres" or backend == "postgresql":
            logger.info("Using PostgreSQL federation registry store")
            try:
                from aragora.storage.postgres_store import get_postgres_pool

                # Initialize PostgreSQL store with connection pool
                pool = asyncio.get_event_loop().run_until_complete(get_postgres_pool())
                store = PostgresFederationRegistryStore(pool)
                asyncio.get_event_loop().run_until_complete(store.initialize())
                _federation_registry_store = store
            except Exception as e:
                logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
                _federation_registry_store = SQLiteFederationRegistryStore()
        else:  # Default to SQLite
            _federation_registry_store = SQLiteFederationRegistryStore()
            logger.info("Using SQLite federation registry store")

        return _federation_registry_store


def set_federation_registry_store(store: FederationRegistryStoreBackend) -> None:
    """Set a custom federation registry store instance."""
    global _federation_registry_store

    with _store_lock:
        _federation_registry_store = store


def reset_federation_registry_store() -> None:
    """Reset the global federation registry store (for testing)."""
    global _federation_registry_store

    with _store_lock:
        _federation_registry_store = None


__all__ = [
    "FederatedRegionConfig",
    "FederationRegistryStoreBackend",
    "InMemoryFederationRegistryStore",
    "SQLiteFederationRegistryStore",
    "RedisFederationRegistryStore",
    "PostgresFederationRegistryStore",
    "get_federation_registry_store",
    "set_federation_registry_store",
    "reset_federation_registry_store",
]
