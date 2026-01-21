"""
Persistent storage for enterprise connector sync state.

Provides SQLite/PostgreSQL-backed storage for:
- Connector configurations (with encrypted credentials)
- Sync job history and status
- Sync statistics and metrics

Usage:
    store = SyncStore()
    await store.initialize()

    # Save connector config
    await store.save_connector(connector_id, config)

    # Record sync job
    await store.record_sync_start(connector_id, sync_id)
    await store.record_sync_complete(sync_id, items_synced, status)

    # Get history
    history = await store.get_sync_history(connector_id, limit=50)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Import encryption (optional - graceful degradation if not available)
try:
    from aragora.security.encryption import get_encryption_service, CRYPTO_AVAILABLE
except ImportError:
    CRYPTO_AVAILABLE = False

    def get_encryption_service():
        raise RuntimeError("Encryption not available")

# Credential fields that should be encrypted
CREDENTIAL_KEYWORDS = frozenset([
    "api_key", "secret", "password", "token", "auth_token",
    "access_key", "private_key", "credentials", "client_secret",
])


def _is_sensitive_key(key: str) -> bool:
    """Check if a config key is sensitive (should be encrypted)."""
    key_lower = key.lower()
    return any(kw in key_lower for kw in CREDENTIAL_KEYWORDS)


def _encrypt_config(
    config: Dict[str, Any], use_encryption: bool, connector_id: str = ""
) -> Dict[str, Any]:
    """
    Encrypt sensitive fields in connector config.

    Uses connector_id as Associated Authenticated Data (AAD) to bind the
    ciphertext to a specific connector, preventing cross-connector attacks.
    """
    if not use_encryption or not CRYPTO_AVAILABLE or not config:
        return config
    try:
        service = get_encryption_service()
        sensitive_keys = [k for k in config if _is_sensitive_key(k)]
        if not sensitive_keys:
            return config
        # AAD binds config to this specific connector
        return service.encrypt_fields(config, sensitive_keys, connector_id if connector_id else None)
    except Exception as e:
        logger.warning(f"Config encryption unavailable for {connector_id}: {e}")
        return config


def _decrypt_config(
    config: Dict[str, Any], use_encryption: bool, connector_id: str = ""
) -> Dict[str, Any]:
    """
    Decrypt sensitive fields in connector config.

    AAD must match what was used during encryption.
    """
    if not use_encryption or not CRYPTO_AVAILABLE or not config:
        return config

    # Check for encryption markers - if none present, it's legacy data
    has_encrypted = any(
        isinstance(v, dict) and v.get("_encrypted")
        for v in config.values()
    )
    if not has_encrypted:
        return config  # Legacy unencrypted data - return as-is

    try:
        service = get_encryption_service()
        sensitive_keys = [k for k in config if isinstance(config[k], dict) and config[k].get("_encrypted")]
        if not sensitive_keys:
            return config
        return service.decrypt_fields(config, sensitive_keys, connector_id if connector_id else None)
    except Exception as e:
        logger.warning(f"Config decryption failed for {connector_id}: {e}")
        return config


@dataclass
class ConnectorConfig:
    """Stored connector configuration."""

    id: str
    connector_type: str
    name: str
    config: Dict[str, Any]
    status: str = "configured"  # configured, active, error, disabled
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_sync_at: Optional[datetime] = None
    last_sync_status: Optional[str] = None
    items_indexed: int = 0
    error_message: Optional[str] = None


@dataclass
class SyncJob:
    """Record of a sync operation."""

    id: str
    connector_id: str
    status: str  # running, completed, failed, cancelled
    started_at: datetime
    completed_at: Optional[datetime] = None
    items_synced: int = 0
    items_failed: int = 0
    error_message: Optional[str] = None
    duration_seconds: Optional[float] = None


class SyncStore:
    """
    Persistent storage backend for connector sync state.

    Supports SQLite (default) and PostgreSQL backends.
    Uses aiosqlite for async SQLite access.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        use_encryption: bool = True,
    ):
        """
        Initialize the sync store.

        Args:
            database_url: Database URL. Defaults to SQLite file.
                - sqlite:///path/to/db.sqlite
                - postgresql://user:pass@host/db
            use_encryption: Whether to encrypt sensitive config fields
        """
        self._database_url = database_url or os.environ.get(
            "ARAGORA_SYNC_DATABASE_URL", "sqlite:///data/connectors.db"
        )
        self._use_encryption = use_encryption
        self._initialized = False
        self._connection = None

        # In-memory cache for fast access
        self._connectors_cache: Dict[str, ConnectorConfig] = {}
        self._active_jobs: Dict[str, SyncJob] = {}

    async def initialize(self) -> None:
        """Initialize database connection and create tables."""
        if self._initialized:
            return

        if self._database_url.startswith("sqlite"):
            await self._init_sqlite()
        elif self._database_url.startswith("postgresql"):
            await self._init_postgres()
        else:
            raise ValueError(f"Unsupported database URL: {self._database_url}")

        self._initialized = True
        logger.info("SyncStore initialized with %s", self._database_url.split("://")[0])

    async def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        try:
            import aiosqlite
        except ImportError:
            logger.warning(
                "CONNECTOR SYNC STORE: aiosqlite not installed - using in-memory fallback. "
                "DATA WILL BE LOST ON RESTART! Install with: pip install aiosqlite"
            )
            return

        # Extract path from URL
        db_path = self._database_url.replace("sqlite:///", "")

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._connection = await aiosqlite.connect(db_path)
        if self._connection is None:
            raise RuntimeError("Database connection failed")

        # Create tables
        await self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS connectors (
                id TEXT PRIMARY KEY,
                connector_type TEXT NOT NULL,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT DEFAULT 'configured',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_sync_at TEXT,
                last_sync_status TEXT,
                items_indexed INTEGER DEFAULT 0,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS sync_jobs (
                id TEXT PRIMARY KEY,
                connector_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                items_synced INTEGER DEFAULT 0,
                items_failed INTEGER DEFAULT 0,
                error_message TEXT,
                duration_seconds REAL,
                FOREIGN KEY (connector_id) REFERENCES connectors(id)
            );

            CREATE INDEX IF NOT EXISTS idx_sync_jobs_connector
                ON sync_jobs(connector_id, started_at DESC);

            CREATE INDEX IF NOT EXISTS idx_connectors_status
                ON connectors(status);
        """
        )
        await self._connection.commit()

        # Load connectors into cache
        async with self._connection.execute("SELECT * FROM connectors") as cursor:
            async for row in cursor:
                connector_id = row[0]
                config = ConnectorConfig(
                    id=connector_id,
                    connector_type=row[1],
                    name=row[2],
                    config=_decrypt_config(json.loads(row[3]), self._use_encryption, connector_id),
                    status=row[4],
                    created_at=datetime.fromisoformat(row[5]),
                    updated_at=datetime.fromisoformat(row[6]),
                    last_sync_at=datetime.fromisoformat(row[7]) if row[7] else None,
                    last_sync_status=row[8],
                    items_indexed=row[9] or 0,
                    error_message=row[10],
                )
                self._connectors_cache[config.id] = config

    async def _init_postgres(self) -> None:
        """Initialize PostgreSQL database."""
        try:
            import asyncpg
        except ImportError:
            logger.warning(
                "CONNECTOR SYNC STORE: asyncpg not installed - using in-memory fallback. "
                "DATA WILL BE LOST ON RESTART! Install with: pip install asyncpg"
            )
            return

        self._connection = await asyncpg.connect(self._database_url)
        if self._connection is None:
            raise RuntimeError("Database connection failed")

        # Create tables
        await self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS connectors (
                id TEXT PRIMARY KEY,
                connector_type TEXT NOT NULL,
                name TEXT NOT NULL,
                config_json JSONB NOT NULL,
                status TEXT DEFAULT 'configured',
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                last_sync_at TIMESTAMPTZ,
                last_sync_status TEXT,
                items_indexed INTEGER DEFAULT 0,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS sync_jobs (
                id TEXT PRIMARY KEY,
                connector_id TEXT NOT NULL REFERENCES connectors(id),
                status TEXT NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                items_synced INTEGER DEFAULT 0,
                items_failed INTEGER DEFAULT 0,
                error_message TEXT,
                duration_seconds REAL
            );

            CREATE INDEX IF NOT EXISTS idx_sync_jobs_connector
                ON sync_jobs(connector_id, started_at DESC);

            CREATE INDEX IF NOT EXISTS idx_connectors_status
                ON connectors(status);
        """
        )

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False

    # ==================== Connector Operations ====================

    async def save_connector(
        self,
        connector_id: str,
        connector_type: str,
        name: str,
        config: Dict[str, Any],
    ) -> ConnectorConfig:
        """
        Save or update a connector configuration.

        Args:
            connector_id: Unique connector ID
            connector_type: Type (github, s3, sharepoint, etc.)
            name: Display name
            config: Configuration dictionary (credentials encrypted if enabled)

        Returns:
            Saved ConnectorConfig
        """
        now = datetime.now(timezone.utc)

        existing = self._connectors_cache.get(connector_id)
        if existing:
            # Update
            existing.name = name
            existing.config = config
            existing.updated_at = now
            connector = existing
        else:
            # Create new
            connector = ConnectorConfig(
                id=connector_id,
                connector_type=connector_type,
                name=name,
                config=config,
                created_at=now,
                updated_at=now,
            )

        self._connectors_cache[connector_id] = connector

        # Persist to database
        if self._connection:
            # Encrypt config with connector_id as AAD for integrity
            encrypted_config = _encrypt_config(config, self._use_encryption, connector_id)
            config_json = json.dumps(encrypted_config)

            if self._database_url.startswith("sqlite"):
                await self._connection.execute(
                    """
                    INSERT OR REPLACE INTO connectors
                    (id, connector_type, name, config_json, status,
                     created_at, updated_at, last_sync_at, last_sync_status,
                     items_indexed, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        connector.id,
                        connector.connector_type,
                        connector.name,
                        config_json,
                        connector.status,
                        connector.created_at.isoformat(),
                        connector.updated_at.isoformat(),
                        connector.last_sync_at.isoformat() if connector.last_sync_at else None,
                        connector.last_sync_status,
                        connector.items_indexed,
                        connector.error_message,
                    ),
                )
                await self._connection.commit()
            else:
                # PostgreSQL
                await self._connection.execute(
                    """
                    INSERT INTO connectors
                    (id, connector_type, name, config_json, status,
                     created_at, updated_at, last_sync_at, last_sync_status,
                     items_indexed, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        config_json = EXCLUDED.config_json,
                        updated_at = EXCLUDED.updated_at
                """,
                    connector.id,
                    connector.connector_type,
                    connector.name,
                    encrypted_config,
                    connector.status,
                    connector.created_at,
                    connector.updated_at,
                    connector.last_sync_at,
                    connector.last_sync_status,
                    connector.items_indexed,
                    connector.error_message,
                )

        return connector

    async def get_connector(self, connector_id: str) -> Optional[ConnectorConfig]:
        """Get connector by ID."""
        return self._connectors_cache.get(connector_id)

    async def list_connectors(
        self,
        status: Optional[str] = None,
        connector_type: Optional[str] = None,
    ) -> List[ConnectorConfig]:
        """List all connectors, optionally filtered."""
        connectors = list(self._connectors_cache.values())

        if status:
            connectors = [c for c in connectors if c.status == status]
        if connector_type:
            connectors = [c for c in connectors if c.connector_type == connector_type]

        return sorted(connectors, key=lambda c: c.updated_at, reverse=True)

    async def delete_connector(self, connector_id: str) -> bool:
        """Delete a connector."""
        if connector_id not in self._connectors_cache:
            return False

        del self._connectors_cache[connector_id]

        if self._connection:
            if self._database_url.startswith("sqlite"):
                await self._connection.execute(
                    "DELETE FROM connectors WHERE id = ?", (connector_id,)
                )
                await self._connection.commit()
            else:
                await self._connection.execute("DELETE FROM connectors WHERE id = $1", connector_id)

        return True

    async def update_connector_status(
        self,
        connector_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update connector status."""
        connector = self._connectors_cache.get(connector_id)
        if not connector:
            return

        connector.status = status
        connector.error_message = error_message
        connector.updated_at = datetime.now(timezone.utc)

        if self._connection:
            if self._database_url.startswith("sqlite"):
                await self._connection.execute(
                    """
                    UPDATE connectors
                    SET status = ?, error_message = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (status, error_message, connector.updated_at.isoformat(), connector_id),
                )
                await self._connection.commit()

    # ==================== Sync Job Operations ====================

    async def record_sync_start(
        self,
        connector_id: str,
        sync_id: Optional[str] = None,
    ) -> SyncJob:
        """Record the start of a sync operation."""
        job = SyncJob(
            id=sync_id or str(uuid4()),
            connector_id=connector_id,
            status="running",
            started_at=datetime.now(timezone.utc),
        )

        self._active_jobs[job.id] = job

        # Update connector status
        if connector_id in self._connectors_cache:
            self._connectors_cache[connector_id].status = "active"

        # Persist
        if self._connection:
            if self._database_url.startswith("sqlite"):
                await self._connection.execute(
                    """
                    INSERT INTO sync_jobs
                    (id, connector_id, status, started_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (job.id, job.connector_id, job.status, job.started_at.isoformat()),
                )
                await self._connection.commit()

        return job

    async def record_sync_progress(
        self,
        sync_id: str,
        items_synced: int,
        items_failed: int = 0,
    ) -> None:
        """Update sync progress."""
        job = self._active_jobs.get(sync_id)
        if job:
            job.items_synced = items_synced
            job.items_failed = items_failed

    async def record_sync_complete(
        self,
        sync_id: str,
        status: str = "completed",
        items_synced: Optional[int] = None,
        items_failed: int = 0,
        error_message: Optional[str] = None,
    ) -> Optional[SyncJob]:
        """Record sync completion."""
        job = self._active_jobs.pop(sync_id, None)
        if not job:
            return None

        job.status = status
        job.completed_at = datetime.now(timezone.utc)
        job.duration_seconds = (job.completed_at - job.started_at).total_seconds()
        if items_synced is not None:
            job.items_synced = items_synced
        job.items_failed = items_failed
        job.error_message = error_message

        # Update connector
        connector = self._connectors_cache.get(job.connector_id)
        if connector:
            connector.status = "configured" if status == "completed" else "error"
            connector.last_sync_at = job.completed_at
            connector.last_sync_status = status
            if status == "completed":
                connector.items_indexed += job.items_synced
                connector.error_message = None
            else:
                connector.error_message = error_message

        # Persist
        if self._connection:
            if self._database_url.startswith("sqlite"):
                await self._connection.execute(
                    """
                    UPDATE sync_jobs
                    SET status = ?, completed_at = ?, items_synced = ?,
                        items_failed = ?, error_message = ?, duration_seconds = ?
                    WHERE id = ?
                """,
                    (
                        job.status,
                        job.completed_at.isoformat(),
                        job.items_synced,
                        job.items_failed,
                        job.error_message,
                        job.duration_seconds,
                        job.id,
                    ),
                )
                await self._connection.commit()

        return job

    async def get_sync_history(
        self,
        connector_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SyncJob]:
        """Get sync history, optionally filtered by connector."""
        jobs = []

        if self._connection and self._database_url.startswith("sqlite"):
            if connector_id:
                query = """
                    SELECT * FROM sync_jobs
                    WHERE connector_id = ?
                    ORDER BY started_at DESC
                    LIMIT ? OFFSET ?
                """
                params = (connector_id, limit, offset)
            else:
                query = """
                    SELECT * FROM sync_jobs
                    ORDER BY started_at DESC
                    LIMIT ? OFFSET ?
                """
                params = (limit, offset)

            async with self._connection.execute(query, params) as cursor:
                async for row in cursor:
                    jobs.append(
                        SyncJob(
                            id=row[0],
                            connector_id=row[1],
                            status=row[2],
                            started_at=datetime.fromisoformat(row[3]),
                            completed_at=datetime.fromisoformat(row[4]) if row[4] else None,
                            items_synced=row[5] or 0,
                            items_failed=row[6] or 0,
                            error_message=row[7],
                            duration_seconds=row[8],
                        )
                    )

        # Include active jobs
        for job in self._active_jobs.values():
            if connector_id is None or job.connector_id == connector_id:
                jobs.append(job)

        return sorted(jobs, key=lambda j: j.started_at, reverse=True)[:limit]

    async def get_sync_stats(
        self,
        connector_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get aggregate sync statistics."""
        stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "total_items_synced": 0,
            "total_items_failed": 0,
            "avg_duration_seconds": 0.0,
            "active_syncs": len(self._active_jobs),
        }

        if self._connection and self._database_url.startswith("sqlite"):
            if connector_id:
                query = """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(items_synced) as items_synced,
                        SUM(items_failed) as items_failed,
                        AVG(duration_seconds) as avg_duration
                    FROM sync_jobs
                    WHERE connector_id = ?
                """
                params = (connector_id,)
            else:
                query = """
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(items_synced) as items_synced,
                        SUM(items_failed) as items_failed,
                        AVG(duration_seconds) as avg_duration
                    FROM sync_jobs
                """
                params = ()

            async with self._connection.execute(query, params) as cursor:
                row = await cursor.fetchone()
                if row:
                    stats["total_syncs"] = row[0] or 0
                    stats["successful_syncs"] = row[1] or 0
                    stats["failed_syncs"] = row[2] or 0
                    stats["total_items_synced"] = row[3] or 0
                    stats["total_items_failed"] = row[4] or 0
                    stats["avg_duration_seconds"] = row[5] or 0.0

        return stats


# Global instance for easy access
_store: Optional[SyncStore] = None


async def get_sync_store() -> SyncStore:
    """Get the global SyncStore instance, initializing if needed."""
    global _store
    if _store is None:
        _store = SyncStore()
        await _store.initialize()
    return _store


__all__ = [
    "SyncStore",
    "SyncJob",
    "ConnectorConfig",
    "get_sync_store",
]
