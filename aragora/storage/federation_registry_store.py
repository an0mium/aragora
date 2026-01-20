"""
Federation Registry Storage Backends.

Persistent storage for federated region configurations used by Knowledge Mound.

Backends:
- InMemoryFederationRegistryStore: For testing
- SQLiteFederationRegistryStore: For single-instance deployments
- RedisFederationRegistryStore: For multi-instance (with SQLite fallback)

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

import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        now = datetime.utcnow().isoformat()
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
    async def get(self, region_id: str, workspace_id: Optional[str] = None) -> Optional[FederatedRegionConfig]:
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

    async def get(self, region_id: str, workspace_id: Optional[str] = None) -> Optional[FederatedRegionConfig]:
        """Get a federated region by ID."""
        key = self._make_key(region_id, workspace_id)
        with self._lock:
            return self._data.get(key)

    async def save(self, region: FederatedRegionConfig) -> None:
        """Save a federated region configuration."""
        region.updated_at = datetime.utcnow().isoformat()
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
                return [
                    r for r in self._data.values()
                    if r.workspace_id == workspace_id
                ]
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
            now = datetime.utcnow().isoformat()
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

    async def get(self, region_id: str, workspace_id: Optional[str] = None) -> Optional[FederatedRegionConfig]:
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
        region.updated_at = datetime.utcnow().isoformat()
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
            now = datetime.utcnow().isoformat()
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

    async def get(self, region_id: str, workspace_id: Optional[str] = None) -> Optional[FederatedRegionConfig]:
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
        region.updated_at = datetime.utcnow().isoformat()

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
            now = datetime.utcnow().isoformat()
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
            except Exception:
                pass


def get_federation_registry_store() -> FederationRegistryStoreBackend:
    """
    Get the global federation registry store instance.

    Backend is selected based on ARAGORA_FEDERATION_STORE_BACKEND env var:
    - "memory": InMemoryFederationRegistryStore (for testing)
    - "sqlite": SQLiteFederationRegistryStore (default, single-instance)
    - "redis": RedisFederationRegistryStore (multi-instance)
    """
    global _federation_registry_store

    with _store_lock:
        if _federation_registry_store is not None:
            return _federation_registry_store

        backend = os.getenv("ARAGORA_FEDERATION_STORE_BACKEND", "sqlite").lower()

        if backend == "memory":
            _federation_registry_store = InMemoryFederationRegistryStore()
            logger.info("Using in-memory federation registry store")
        elif backend == "redis":
            _federation_registry_store = RedisFederationRegistryStore()
            logger.info("Using Redis federation registry store")
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
    "get_federation_registry_store",
    "set_federation_registry_store",
    "reset_federation_registry_store",
]
