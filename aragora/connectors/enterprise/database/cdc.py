"""
Change Data Capture (CDC) utilities.

Provides a unified model for tracking database changes across
different database systems (PostgreSQL, MongoDB, etc.).

Features:
- Unified ChangeEvent model for all operations
- Resume token management for reliable streaming
- Change event handlers for Knowledge Mound integration
- Operation type classification
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.knowledge.mound.core import KnowledgeMound

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class ChangeOperation(str, Enum):
    """Type of database change operation."""

    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    REPLACE = "replace"  # MongoDB-specific
    UPSERT = "upsert"
    TRUNCATE = "truncate"
    SCHEMA_CHANGE = "schema_change"


class CDCSourceType(str, Enum):
    """Source database type for CDC events."""

    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    SNOWFLAKE = "snowflake"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class ChangeEvent:
    """
    Unified representation of a database change event.

    Works across PostgreSQL (LISTEN/NOTIFY, logical replication),
    MongoDB (change streams), and other databases.
    """

    # Event identification
    id: str  # Unique event ID
    source_type: CDCSourceType
    connector_id: str

    # Operation details
    operation: ChangeOperation
    timestamp: datetime

    # Location
    database: str
    schema: Optional[str] = None  # PostgreSQL schema or MongoDB database
    table: str = ""  # Table or collection name

    # Key identification
    primary_key: Optional[Dict[str, Any]] = None  # Primary key values
    document_id: Optional[str] = None  # MongoDB _id or row identifier

    # Data (for insert/update/replace)
    data: Optional[Dict[str, Any]] = None  # New data after change
    old_data: Optional[Dict[str, Any]] = None  # Old data before change (if available)

    # Change tracking
    fields_changed: List[str] = field(default_factory=list)

    # Resume support
    resume_token: Optional[str] = None  # MongoDB resume token or LSN
    sequence_number: Optional[int] = None  # For ordering

    # Metadata
    transaction_id: Optional[str] = None
    user: Optional[str] = None  # User who made the change
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique event ID."""
        components = [
            self.source_type.value,
            self.database,
            self.table,
            self.operation.value,
            self.timestamp.isoformat() if self.timestamp else "",
            str(self.primary_key or self.document_id or ""),
        ]
        content = ":".join(components)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @property
    def is_data_change(self) -> bool:
        """Check if this event represents a data change (not schema change)."""
        return self.operation in (
            ChangeOperation.INSERT,
            ChangeOperation.UPDATE,
            ChangeOperation.DELETE,
            ChangeOperation.REPLACE,
            ChangeOperation.UPSERT,
        )

    @property
    def qualified_table(self) -> str:
        """Get fully qualified table name."""
        if self.schema:
            return f"{self.schema}.{self.table}"
        return self.table

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source_type": self.source_type.value,
            "connector_id": self.connector_id,
            "operation": self.operation.value,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "database": self.database,
            "schema": self.schema,
            "table": self.table,
            "primary_key": self.primary_key,
            "document_id": self.document_id,
            "data": self.data,
            "old_data": self.old_data,
            "fields_changed": self.fields_changed,
            "resume_token": self.resume_token,
            "sequence_number": self.sequence_number,
            "transaction_id": self.transaction_id,
            "user": self.user,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChangeEvent":
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            id=data.get("id", ""),
            source_type=CDCSourceType(data.get("source_type", "postgresql")),
            connector_id=data.get("connector_id", ""),
            operation=ChangeOperation(data.get("operation", "update")),
            timestamp=timestamp or datetime.now(timezone.utc),
            database=data.get("database", ""),
            schema=data.get("schema"),
            table=data.get("table", ""),
            primary_key=data.get("primary_key"),
            document_id=data.get("document_id"),
            data=data.get("data"),
            old_data=data.get("old_data"),
            fields_changed=data.get("fields_changed", []),
            resume_token=data.get("resume_token"),
            sequence_number=data.get("sequence_number"),
            transaction_id=data.get("transaction_id"),
            user=data.get("user"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_postgres_notify(
        cls,
        payload: str,
        channel: str,
        connector_id: str,
        database: str,
        schema: str = "public",
    ) -> "ChangeEvent":
        """Create ChangeEvent from PostgreSQL NOTIFY payload."""
        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {"raw": payload}

        operation_map = {
            "INSERT": ChangeOperation.INSERT,
            "UPDATE": ChangeOperation.UPDATE,
            "DELETE": ChangeOperation.DELETE,
            "TRUNCATE": ChangeOperation.TRUNCATE,
        }

        return cls(
            id="",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id=connector_id,
            operation=operation_map.get(
                data.get("operation", "UPDATE").upper(), ChangeOperation.UPDATE
            ),
            timestamp=datetime.now(timezone.utc),
            database=database,
            schema=schema,
            table=data.get("table", channel),
            primary_key=data.get("primary_key"),
            data=data.get("new_data") or data.get("data"),
            old_data=data.get("old_data"),
            fields_changed=data.get("changed_columns", []),
            transaction_id=data.get("xid"),
            user=data.get("user"),
            metadata={"channel": channel, "raw_payload": payload},
        )

    @classmethod
    def from_mongodb_change(
        cls,
        change: Dict[str, Any],
        connector_id: str,
    ) -> "ChangeEvent":
        """Create ChangeEvent from MongoDB change stream document."""
        operation_map = {
            "insert": ChangeOperation.INSERT,
            "update": ChangeOperation.UPDATE,
            "replace": ChangeOperation.REPLACE,
            "delete": ChangeOperation.DELETE,
        }

        ns = change.get("ns", {})
        document_key = change.get("documentKey", {})
        full_document = change.get("fullDocument")
        update_description = change.get("updateDescription", {})

        # Extract document ID
        doc_id = document_key.get("_id")
        if doc_id and hasattr(doc_id, "__str__"):
            doc_id = str(doc_id)

        # Get changed fields from update description
        fields_changed = []
        if update_description:
            updated = update_description.get("updatedFields", {})
            removed = update_description.get("removedFields", [])
            fields_changed = list(updated.keys()) + removed

        # Get resume token
        resume_token = change.get("_id")
        if resume_token:
            resume_token = json.dumps(resume_token, default=str)

        return cls(
            id="",
            source_type=CDCSourceType.MONGODB,
            connector_id=connector_id,
            operation=operation_map.get(
                change.get("operationType", "update"), ChangeOperation.UPDATE
            ),
            timestamp=change.get("clusterTime", datetime.now(timezone.utc)),
            database=ns.get("db", ""),
            table=ns.get("coll", ""),
            document_id=doc_id,
            data=full_document,
            old_data=change.get("fullDocumentBeforeChange"),
            fields_changed=fields_changed,
            resume_token=resume_token,
            transaction_id=str(change.get("txnNumber")) if change.get("txnNumber") else None,
            metadata={"operation_type": change.get("operationType")},
        )


# =============================================================================
# Resume Token Management
# =============================================================================


@dataclass
class ResumeToken:
    """
    Persistent resume token for CDC streams.

    Enables resuming change streams from the last processed position
    after connector restarts.
    """

    connector_id: str
    source_type: CDCSourceType
    token: str  # Opaque token (MongoDB resume token, PostgreSQL LSN, etc.)
    timestamp: datetime
    sequence_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "source_type": self.source_type.value,
            "token": self.token,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResumeToken":
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        return cls(
            connector_id=data.get("connector_id", ""),
            source_type=CDCSourceType(data.get("source_type", "postgresql")),
            token=data.get("token", ""),
            timestamp=timestamp or datetime.now(timezone.utc),
            sequence_number=data.get("sequence_number"),
            metadata=data.get("metadata", {}),
        )


class ResumeTokenStore:
    """
    Persistent storage for CDC resume tokens.

    Stores tokens in a JSON file for simple file-based persistence.
    Can be extended for database-backed storage.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = Path.home() / ".aragora" / "cdc_resume_tokens.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokens: Dict[str, ResumeToken] = {}
        self._load()

    def _load(self) -> None:
        """Load tokens from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self._tokens[key] = ResumeToken.from_dict(value)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load resume tokens: {e}")
                self._tokens = {}

    def _save(self) -> None:
        """Save tokens to storage."""
        try:
            with open(self.storage_path, "w") as f:
                data = {key: token.to_dict() for key, token in self._tokens.items()}
                json.dump(data, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save resume tokens: {e}")

    def get(self, connector_id: str) -> Optional[ResumeToken]:
        """Get resume token for a connector."""
        return self._tokens.get(connector_id)

    def save(self, token: ResumeToken) -> None:
        """Save a resume token."""
        self._tokens[token.connector_id] = token
        self._save()

    def delete(self, connector_id: str) -> None:
        """Delete a resume token."""
        if connector_id in self._tokens:
            del self._tokens[connector_id]
            self._save()

    def clear_all(self) -> None:
        """Clear all resume tokens."""
        self._tokens = {}
        self._save()


# =============================================================================
# Change Event Handlers
# =============================================================================


class ChangeEventHandler(ABC):
    """Abstract handler for processing change events."""

    @abstractmethod
    async def handle(self, event: ChangeEvent) -> bool:
        """
        Handle a change event.

        Returns True if event was processed successfully.
        """
        pass


class KnowledgeMoundHandler(ChangeEventHandler):
    """
    Handler that pushes change events to the Knowledge Mound.

    Integrates CDC with the knowledge management system for
    real-time knowledge updates.
    """

    def __init__(
        self,
        workspace_id: str = "default",
        auto_ingest: bool = True,
        delete_on_remove: bool = True,
    ):
        self.workspace_id = workspace_id
        self.auto_ingest = auto_ingest
        self.delete_on_remove = delete_on_remove
        self._mound: Optional["KnowledgeMound"] = None

    async def _get_mound(self) -> "KnowledgeMound":
        """Get or create Knowledge Mound instance."""
        if self._mound is None:
            from aragora.knowledge.mound.core import KnowledgeMound

            self._mound = KnowledgeMound(workspace_id=self.workspace_id)
        return self._mound

    async def handle(self, event: ChangeEvent) -> bool:
        """Process change event and update Knowledge Mound."""
        try:
            if not event.is_data_change:
                logger.debug(f"Skipping non-data event: {event.operation}")
                return True

            mound = await self._get_mound()

            if event.operation == ChangeOperation.DELETE:
                if self.delete_on_remove:
                    # Mark knowledge as outdated/removed
                    await self._handle_delete(mound, event)
            else:
                if self.auto_ingest:
                    await self._handle_upsert(mound, event)

            return True

        except Exception as e:
            logger.error(f"Failed to handle change event: {e}")
            return False

    async def _handle_upsert(self, mound: "KnowledgeMound", event: ChangeEvent) -> None:
        """Handle insert/update/replace events."""
        from aragora.knowledge.mound.ingestion import IngestionRequest
        from aragora.knowledge.mound.types import KnowledgeSource

        if not event.data:
            logger.debug(f"No data in event {event.id}, skipping ingestion")
            return

        # Convert data to text content
        content = self._data_to_content(event.data)
        if not content:
            return

        request = IngestionRequest(
            content=content,
            workspace_id=self.workspace_id,
            source_type=KnowledgeSource.FACT,
            confidence=0.8,
            topics=[event.table],
            metadata={
                "source": event.source_type.value,
                "database": event.database,
                "table": event.table,
                "document_id": event.document_id or str(event.primary_key),
                "operation": event.operation.value,
                "timestamp": event.timestamp.isoformat(),
            },
        )

        await mound.store(request)
        logger.debug(f"Ingested change event {event.id} to Knowledge Mound")

    async def _handle_delete(self, mound: "KnowledgeMound", event: ChangeEvent) -> None:
        """Handle delete events."""
        # Mark the knowledge as outdated by searching and updating
        doc_id = event.document_id or str(event.primary_key)
        logger.info(f"Document deleted: {event.table}/{doc_id}")
        # Note: Full deletion support would require Knowledge Mound delete API

    def _data_to_content(self, data: Dict[str, Any]) -> str:
        """Convert document data to text content."""
        parts = []
        for key, value in data.items():
            if key.startswith("_"):
                continue
            if value is not None:
                if isinstance(value, datetime):
                    value = value.isoformat()
                elif isinstance(value, (dict, list)):
                    value = json.dumps(value, default=str)
                parts.append(f"{key}: {value}")
        return "\n".join(parts)

    def _extract_title(self, event: ChangeEvent) -> str:
        """Extract a title from the event."""
        if event.data:
            for field in ["title", "name", "subject", "label"]:
                if event.data.get(field):
                    return str(event.data[field])[:100]

        doc_id = event.document_id or str(event.primary_key or "")
        return f"{event.table}/{doc_id[:20]}"


class CallbackHandler(ChangeEventHandler):
    """Handler that calls a callback function for each event."""

    def __init__(self, callback: Callable[[ChangeEvent], bool]):
        self.callback = callback

    async def handle(self, event: ChangeEvent) -> bool:
        """Call the callback with the event."""
        if asyncio.iscoroutinefunction(self.callback):
            return await self.callback(event)
        return self.callback(event)


class CompositeHandler(ChangeEventHandler):
    """Handler that delegates to multiple handlers."""

    def __init__(self, handlers: Optional[List[ChangeEventHandler]] = None):
        self.handlers = handlers or []

    def add_handler(self, handler: ChangeEventHandler) -> None:
        """Add a handler."""
        self.handlers.append(handler)

    async def handle(self, event: ChangeEvent) -> bool:
        """Handle event with all handlers."""
        results = []
        for handler in self.handlers:
            try:
                result = await handler.handle(event)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed: {e}")
                results.append(False)

        return all(results)


# =============================================================================
# CDC Stream Manager
# =============================================================================


class CDCStreamManager:
    """
    Manages CDC streams for a connector.

    Coordinates:
    - Resume token persistence
    - Event handler dispatch
    - Stream lifecycle (start/stop)
    - Error handling and recovery
    """

    def __init__(
        self,
        connector_id: str,
        source_type: CDCSourceType,
        handler: Optional[ChangeEventHandler] = None,
        token_store: Optional[ResumeTokenStore] = None,
    ):
        self.connector_id = connector_id
        self.source_type = source_type
        self.handler = handler or CompositeHandler()
        self.token_store = token_store or ResumeTokenStore()

        self._running = False
        self._events_processed = 0
        self._last_event_time: Optional[datetime] = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        """Get stream statistics."""
        return {
            "connector_id": self.connector_id,
            "source_type": self.source_type.value,
            "running": self._running,
            "events_processed": self._events_processed,
            "last_event_time": self._last_event_time.isoformat() if self._last_event_time else None,
        }

    def get_resume_token(self) -> Optional[str]:
        """Get the last saved resume token."""
        token = self.token_store.get(self.connector_id)
        return token.token if token else None

    async def process_event(self, event: ChangeEvent) -> bool:
        """Process a single change event."""
        try:
            success = await self.handler.handle(event)

            if success:
                self._events_processed += 1
                self._last_event_time = event.timestamp

                # Save resume token if available
                if event.resume_token:
                    self.token_store.save(
                        ResumeToken(
                            connector_id=self.connector_id,
                            source_type=self.source_type,
                            token=event.resume_token,
                            timestamp=event.timestamp,
                            sequence_number=event.sequence_number,
                        )
                    )

            return success

        except Exception as e:
            logger.error(f"Failed to process event {event.id}: {e}")
            return False

    def start(self) -> None:
        """Mark stream as running."""
        self._running = True
        logger.info(f"CDC stream started for {self.connector_id}")

    def stop(self) -> None:
        """Mark stream as stopped."""
        self._running = False
        logger.info(f"CDC stream stopped for {self.connector_id}")

    def reset(self) -> None:
        """Reset stream state."""
        self._events_processed = 0
        self._last_event_time = None
        self.token_store.delete(self.connector_id)
