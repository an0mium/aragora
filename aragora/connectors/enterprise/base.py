"""
Enterprise Connector Base Class.

Extends BaseConnector with incremental sync capabilities for
enterprise data source integration with the Knowledge Mound.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Protocol

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Status of a sync operation."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SyncState:
    """
    Persistent state for incremental sync operations.

    Tracks cursor/token position to enable resumable syncs.
    """

    connector_id: str
    tenant_id: str = "default"

    # Cursor tracking
    cursor: Optional[str] = None  # Opaque cursor for pagination
    last_sync_at: Optional[datetime] = None
    last_item_id: Optional[str] = None  # Last processed item ID
    last_item_timestamp: Optional[datetime] = None

    # Sync metadata
    items_synced: int = 0
    items_total: int = 0
    errors: List[str] = field(default_factory=list)

    # Status
    status: SyncStatus = SyncStatus.IDLE
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "connector_id": self.connector_id,
            "tenant_id": self.tenant_id,
            "cursor": self.cursor,
            "last_sync_at": self.last_sync_at.isoformat() if self.last_sync_at else None,
            "last_item_id": self.last_item_id,
            "last_item_timestamp": self.last_item_timestamp.isoformat() if self.last_item_timestamp else None,
            "items_synced": self.items_synced,
            "items_total": self.items_total,
            "errors": self.errors[-10:],  # Keep last 10 errors
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncState":
        """Deserialize from dictionary."""
        return cls(
            connector_id=data["connector_id"],
            tenant_id=data.get("tenant_id", "default"),
            cursor=data.get("cursor"),
            last_sync_at=datetime.fromisoformat(data["last_sync_at"]) if data.get("last_sync_at") else None,
            last_item_id=data.get("last_item_id"),
            last_item_timestamp=datetime.fromisoformat(data["last_item_timestamp"]) if data.get("last_item_timestamp") else None,
            items_synced=data.get("items_synced", 0),
            items_total=data.get("items_total", 0),
            errors=data.get("errors", []),
            status=SyncStatus(data.get("status", "idle")),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
        )

    def save(self, path: Path) -> None:
        """Save state to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["SyncState"]:
        """Load state from file."""
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Failed to load sync state from {path}: {e}")
            return None


@dataclass
class SyncResult:
    """Result of a sync operation."""

    connector_id: str
    success: bool
    items_synced: int
    items_updated: int
    items_skipped: int
    items_failed: int
    duration_ms: float
    errors: List[str] = field(default_factory=list)
    new_cursor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "connector_id": self.connector_id,
            "success": self.success,
            "items_synced": self.items_synced,
            "items_updated": self.items_updated,
            "items_skipped": self.items_skipped,
            "items_failed": self.items_failed,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "new_cursor": self.new_cursor,
        }


class CredentialProvider(Protocol):
    """Protocol for credential providers."""

    async def get_credential(self, key: str) -> Optional[str]:
        """Get a credential by key."""
        ...

    async def set_credential(self, key: str, value: str) -> None:
        """Set a credential."""
        ...


class EnvCredentialProvider:
    """Credential provider using environment variables."""

    def __init__(self, prefix: str = "ARAGORA_"):
        self.prefix = prefix

    async def get_credential(self, key: str) -> Optional[str]:
        """Get credential from environment variable."""
        env_key = f"{self.prefix}{key.upper()}"
        return os.environ.get(env_key)

    async def set_credential(self, key: str, value: str) -> None:
        """Set credential as environment variable (in-memory only)."""
        env_key = f"{self.prefix}{key.upper()}"
        os.environ[env_key] = value


@dataclass
class SyncItem:
    """An item to be synced to the Knowledge Mound."""

    id: str
    content: str
    source_type: str
    source_id: str
    title: str = ""
    url: str = ""
    author: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    domain: str = "general"
    confidence: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_evidence(self, connector_source_type: SourceType) -> Evidence:
        """Convert to Evidence for compatibility with base connector."""
        return Evidence(
            id=self.id,
            source_type=connector_source_type,
            source_id=self.source_id,
            content=self.content,
            title=self.title,
            url=self.url,
            author=self.author,
            created_at=self.created_at.isoformat() if self.created_at else None,
            confidence=self.confidence,
            metadata=self.metadata,
        )


class EnterpriseConnector(BaseConnector):
    """
    Extended connector for enterprise data sources.

    Provides:
    - Incremental sync with cursor/token tracking
    - Credential management
    - Knowledge Mound ingestion
    - Multi-tenant isolation
    - Webhook support for real-time sync

    Subclasses must implement:
    - source_type: The SourceType for this connector
    - name: Human-readable name
    - sync_items(): Async generator yielding SyncItems
    - search(): Search for evidence (inherited from BaseConnector)
    - fetch(): Fetch specific evidence (inherited from BaseConnector)
    """

    def __init__(
        self,
        connector_id: str,
        tenant_id: str = "default",
        credentials: Optional[CredentialProvider] = None,
        state_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.connector_id = connector_id
        self.tenant_id = tenant_id
        self.credentials = credentials or EnvCredentialProvider()
        self.state_dir = state_dir or Path.home() / ".aragora" / "sync_state"

        # Sync state
        self._state: Optional[SyncState] = None
        self._is_syncing = False
        self._cancel_requested = False

        # Callbacks
        self._on_item_synced: Optional[Callable[[SyncItem], None]] = None
        self._on_progress: Optional[Callable[[int, int], None]] = None

    @property
    def state_path(self) -> Path:
        """Path to sync state file."""
        return self.state_dir / f"{self.connector_id}_{self.tenant_id}.json"

    def load_state(self) -> SyncState:
        """Load or create sync state."""
        if self._state is None:
            self._state = SyncState.load(self.state_path)
            if self._state is None:
                self._state = SyncState(
                    connector_id=self.connector_id,
                    tenant_id=self.tenant_id,
                )
        return self._state

    def save_state(self) -> None:
        """Persist sync state."""
        if self._state:
            self._state.save(self.state_path)

    @abstractmethod
    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield items to sync incrementally.

        Subclasses implement this to fetch items from their source.
        Should use state.cursor for pagination and update state.last_item_id.

        Args:
            state: Current sync state with cursor
            batch_size: Number of items per batch

        Yields:
            SyncItem objects to ingest into Knowledge Mound
        """
        yield  # type: ignore

    async def sync(
        self,
        full_sync: bool = False,
        batch_size: int = 100,
        max_items: Optional[int] = None,
    ) -> SyncResult:
        """
        Run incremental sync from this connector to Knowledge Mound.

        Args:
            full_sync: If True, ignore cursor and sync everything
            batch_size: Items per batch
            max_items: Maximum items to sync (for testing)

        Returns:
            SyncResult with statistics
        """
        if self._is_syncing:
            return SyncResult(
                connector_id=self.connector_id,
                success=False,
                items_synced=0,
                items_updated=0,
                items_skipped=0,
                items_failed=0,
                duration_ms=0,
                errors=["Sync already in progress"],
            )

        self._is_syncing = True
        self._cancel_requested = False
        start_time = time.time()

        state = self.load_state()
        if full_sync:
            state.cursor = None
            state.last_item_id = None

        state.status = SyncStatus.RUNNING
        state.started_at = datetime.now(timezone.utc)
        state.errors = []

        items_synced = 0
        items_updated = 0
        items_skipped = 0
        items_failed = 0
        errors = []

        try:
            async for item in self.sync_items(state, batch_size):
                if self._cancel_requested:
                    logger.info(f"[{self.name}] Sync cancelled")
                    state.status = SyncStatus.CANCELLED
                    break

                if max_items and items_synced >= max_items:
                    logger.info(f"[{self.name}] Reached max items limit: {max_items}")
                    break

                try:
                    result = await self._ingest_item(item)
                    if result == "created":
                        items_synced += 1
                    elif result == "updated":
                        items_updated += 1
                    elif result == "skipped":
                        items_skipped += 1

                    state.last_item_id = item.id
                    if item.updated_at:
                        state.last_item_timestamp = item.updated_at
                    state.items_synced = items_synced + items_updated

                    if self._on_item_synced:
                        self._on_item_synced(item)

                    if self._on_progress:
                        self._on_progress(items_synced + items_updated, state.items_total)

                except Exception as e:
                    items_failed += 1
                    error_msg = f"Failed to ingest {item.id}: {e}"
                    errors.append(error_msg)
                    state.errors.append(error_msg)
                    logger.warning(f"[{self.name}] {error_msg}")

                # Periodic state save
                if (items_synced + items_updated) % 100 == 0:
                    self.save_state()

            if state.status != SyncStatus.CANCELLED:
                state.status = SyncStatus.COMPLETED

        except Exception as e:
            state.status = SyncStatus.FAILED
            errors.append(f"Sync failed: {e}")
            logger.error(f"[{self.name}] Sync failed: {e}")

        finally:
            state.completed_at = datetime.now(timezone.utc)
            state.last_sync_at = datetime.now(timezone.utc)
            self.save_state()
            self._is_syncing = False

        duration_ms = (time.time() - start_time) * 1000

        return SyncResult(
            connector_id=self.connector_id,
            success=state.status == SyncStatus.COMPLETED,
            items_synced=items_synced,
            items_updated=items_updated,
            items_skipped=items_skipped,
            items_failed=items_failed,
            duration_ms=duration_ms,
            errors=errors,
            new_cursor=state.cursor,
        )

    async def _ingest_item(self, item: SyncItem) -> str:
        """
        Ingest a sync item into the Knowledge Mound.

        Returns: "created", "updated", or "skipped"
        """
        try:
            from aragora.knowledge.mound import KnowledgeMound, IngestionRequest, KnowledgeSource

            # Map source type string to enum
            source_type_map = {
                "code": KnowledgeSource.FACT,
                "document": KnowledgeSource.FACT,
                "issue": KnowledgeSource.FACT,
                "pr": KnowledgeSource.FACT,
                "discussion": KnowledgeSource.CONSENSUS,
                "comment": KnowledgeSource.FACT,
            }
            source_type = source_type_map.get(item.source_type, KnowledgeSource.FACT)

            request = IngestionRequest(
                content=item.content,
                workspace_id=self.tenant_id,
                source_type=source_type,
                document_id=item.source_id,
                confidence=item.confidence,
                topics=[item.domain],
                metadata={
                    "connector_id": self.connector_id,
                    "source_url": item.url,
                    "author": item.author,
                    "title": item.title,
                    **item.metadata,
                },
            )

            mound = KnowledgeMound(workspace_id=self.tenant_id)
            await mound.initialize()
            result = await mound.store(request)

            if result.deduplicated:
                return "skipped"
            return "created"

        except ImportError:
            logger.debug(f"[{self.name}] Knowledge Mound not available, skipping ingestion")
            return "skipped"

    def cancel_sync(self) -> None:
        """Request cancellation of running sync."""
        self._cancel_requested = True

    def on_item_synced(self, callback: Callable[[SyncItem], None]) -> None:
        """Register callback for when an item is synced."""
        self._on_item_synced = callback

    def on_progress(self, callback: Callable[[int, int], None]) -> None:
        """Register callback for progress updates."""
        self._on_progress = callback

    async def get_sync_status(self) -> Dict[str, Any]:
        """Get current sync status."""
        state = self.load_state()
        return {
            "connector_id": self.connector_id,
            "connector_name": self.name,
            "tenant_id": self.tenant_id,
            "status": state.status.value,
            "is_syncing": self._is_syncing,
            "last_sync_at": state.last_sync_at.isoformat() if state.last_sync_at else None,
            "items_synced": state.items_synced,
            "cursor": state.cursor,
            "errors": state.errors[-5:],
        }

    # Webhook support for real-time sync

    async def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """
        Handle incoming webhook for real-time sync.

        Subclasses override to handle source-specific webhooks.

        Args:
            payload: Webhook payload

        Returns:
            True if handled successfully
        """
        logger.debug(f"[{self.name}] Webhook received but not implemented")
        return False

    def get_webhook_secret(self) -> Optional[str]:
        """Get webhook secret for signature verification."""
        return None

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        secret = self.get_webhook_secret()
        if not secret:
            return True  # No secret configured, skip verification

        import hmac

        expected = hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signature)
