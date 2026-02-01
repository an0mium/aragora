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
from typing import Any, Callable, Optional, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

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
    schema: str | None = None  # PostgreSQL schema or MongoDB database
    table: str = ""  # Table or collection name

    # Key identification
    primary_key: Optional[dict[str, Any]] = None  # Primary key values
    document_id: str | None = None  # MongoDB _id or row identifier

    # Data (for insert/update/replace)
    data: Optional[dict[str, Any]] = None  # New data after change
    old_data: Optional[dict[str, Any]] = None  # Old data before change (if available)

    # Change tracking
    fields_changed: list[str] = field(default_factory=list)

    # Resume support
    resume_token: str | None = None  # MongoDB resume token or LSN
    sequence_number: int | None = None  # For ordering

    # Metadata
    transaction_id: str | None = None
    user: str | None = None  # User who made the change
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
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

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "ChangeEvent":
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
        change: dict[str, Any],
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
    sequence_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_id": self.connector_id,
            "source_type": self.source_type.value,
            "token": self.token,
            "timestamp": self.timestamp.isoformat(),
            "sequence_number": self.sequence_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResumeToken":
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

    def __init__(self, storage_path: Path | None = None):
        if storage_path:
            self.storage_path = storage_path
        else:
            self.storage_path = Path.home() / ".aragora" / "cdc_resume_tokens.json"
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokens: dict[str, ResumeToken] = {}
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

    def get(self, connector_id: str) -> ResumeToken | None:
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
            from aragora.knowledge.mound import KnowledgeMound

            self._mound = KnowledgeMound(workspace_id=self.workspace_id) # type: ignore[abstract]
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

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            logger.error(f"Failed to handle change event: {e}")
            return False

    async def _handle_upsert(self, mound: "KnowledgeMound", event: ChangeEvent) -> None:
        """Handle insert/update/replace events."""
        from aragora.knowledge.mound.types import IngestionRequest, KnowledgeSource

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

        store = cast(Any, mound).store
        await store(request)
        logger.debug(f"Ingested change event {event.id} to Knowledge Mound")

    async def _handle_delete(self, mound: "KnowledgeMound", event: ChangeEvent) -> None:
        """Handle delete events."""
        # Mark the knowledge as outdated by searching and updating
        doc_id = event.document_id or str(event.primary_key)
        logger.info(f"Document deleted: {event.table}/{doc_id}")
        # Note: Full deletion support would require Knowledge Mound delete API

    def _data_to_content(self, data: dict[str, Any]) -> str:
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
            result = await self.callback(event)
            return bool(result)
        return bool(self.callback(event))


class CompositeHandler(ChangeEventHandler):
    """Handler that delegates to multiple handlers."""

    def __init__(self, handlers: Optional[list[ChangeEventHandler]] = None):
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
            except (ValueError, KeyError, TypeError, RuntimeError) as e:
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
        handler: ChangeEventHandler | None = None,
        token_store: ResumeTokenStore | None = None,
    ):
        self.connector_id = connector_id
        self.source_type = source_type
        self.handler = handler or CompositeHandler()
        self.token_store = token_store or ResumeTokenStore()

        self._running = False
        self._events_processed = 0
        self._last_event_time: datetime | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict[str, Any]:
        """Get stream statistics."""
        return {
            "connector_id": self.connector_id,
            "source_type": self.source_type.value,
            "running": self._running,
            "events_processed": self._events_processed,
            "last_event_time": self._last_event_time.isoformat() if self._last_event_time else None,
        }

    def get_resume_token(self) -> str | None:
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

        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as e:
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


# =============================================================================
# Database CDC Handlers
# =============================================================================


@dataclass
class CDCConfig:
    """
    Configuration for CDC handlers.

    Provides common settings for retry behavior, reconnection,
    and event filtering.
    """

    # Reconnection settings
    max_retries: int = 5
    retry_delay_seconds: float = 1.0
    max_retry_delay_seconds: float = 60.0
    retry_backoff_multiplier: float = 2.0

    # Event filtering
    include_tables: list[str] | None = None  # None = all tables
    exclude_tables: list[str] | None = None
    include_operations: list[ChangeOperation] | None = None  # None = all operations

    # Processing settings
    batch_size: int = 100
    commit_interval_seconds: float = 5.0

    # Health monitoring
    health_check_interval_seconds: float = 30.0


class BaseCDCHandler(ABC):
    """
    Abstract base class for database-specific CDC handlers.

    Provides common functionality for:
    - Connection management with retry logic
    - Event filtering
    - Health monitoring
    - Graceful shutdown
    """

    def __init__(
        self,
        connector_id: str,
        source_type: CDCSourceType,
        config: CDCConfig | None = None,
        event_handler: ChangeEventHandler | None = None,
        token_store: ResumeTokenStore | None = None,
    ):
        self.connector_id = connector_id
        self.source_type = source_type
        self.config = config or CDCConfig()
        self.event_handler = event_handler
        self.token_store = token_store or ResumeTokenStore()

        self._running = False
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._events_processed = 0
        self._last_event_time: datetime | None = None
        self._errors: list[str] = []
        self._connected = False

    @property
    def is_running(self) -> bool:
        """Check if the CDC handler is running."""
        return self._running

    @property
    def is_connected(self) -> bool:
        """Check if connected to the database."""
        return self._connected

    @property
    def stats(self) -> dict[str, Any]:
        """Get handler statistics."""
        return {
            "connector_id": self.connector_id,
            "source_type": self.source_type.value,
            "running": self._running,
            "connected": self._connected,
            "events_processed": self._events_processed,
            "last_event_time": (
                self._last_event_time.isoformat() if self._last_event_time else None
            ),
            "errors": self._errors[-10:],  # Last 10 errors
        }

    def get_resume_token(self) -> str | None:
        """Get the last saved resume token."""
        token = self.token_store.get(self.connector_id)
        return token.token if token else None

    def _should_process_event(self, event: ChangeEvent) -> bool:
        """Check if an event should be processed based on filter config."""
        # Check table filters
        if self.config.include_tables:
            if event.table not in self.config.include_tables:
                return False

        if self.config.exclude_tables:
            if event.table in self.config.exclude_tables:
                return False

        # Check operation filter
        if self.config.include_operations:
            if event.operation not in self.config.include_operations:
                return False

        return True

    async def _process_event(self, event: ChangeEvent) -> bool:
        """
        Process a single change event through the handler chain.

        Returns True if successfully processed.
        """
        if not self._should_process_event(event):
            logger.debug(f"Skipping filtered event: {event.table}/{event.operation.value}")
            return True

        try:
            if self.event_handler:
                success = await self.event_handler.handle(event)
            else:
                # No handler configured - just log
                logger.info(f"CDC event: {event.operation.value} on {event.qualified_table}")
                success = True

            if success:
                self._events_processed += 1
                self._last_event_time = event.timestamp

                # Persist resume token if available
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

        except (ValueError, KeyError, TypeError, RuntimeError) as e:
            error_msg = f"Failed to process event {event.id}: {e}"
            logger.error(error_msg)
            self._errors.append(error_msg)
            return False

    async def _retry_with_backoff(
        self,
        operation: Callable[[], Any],
        operation_name: str = "operation",
    ) -> Any:
        """Execute an operation with exponential backoff retry."""
        delay = self.config.retry_delay_seconds

        for attempt in range(self.config.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation()
                return operation()

            except (OSError, ConnectionError, asyncio.TimeoutError) as e:
                if attempt == self.config.max_retries:
                    logger.error(f"{operation_name} failed after {attempt + 1} attempts: {e}")
                    raise

                logger.warning(
                    f"{operation_name} attempt {attempt + 1}/{self.config.max_retries + 1} "
                    f"failed: {e}. Retrying in {delay:.1f}s"
                )
                await asyncio.sleep(delay)

                # Apply backoff
                delay = min(
                    delay * self.config.retry_backoff_multiplier,
                    self.config.max_retry_delay_seconds,
                )

    @abstractmethod
    async def _connect(self) -> None:
        """Establish connection to the database. Override in subclasses."""
        pass

    @abstractmethod
    async def _disconnect(self) -> None:
        """Close connection to the database. Override in subclasses."""
        pass

    @abstractmethod
    async def _listen_loop(self) -> None:
        """Main loop for listening to CDC events. Override in subclasses."""
        pass

    async def start(self) -> None:
        """Start the CDC handler."""
        if self._running:
            logger.warning(f"CDC handler {self.connector_id} is already running")
            return

        logger.info(f"Starting CDC handler: {self.connector_id}")
        self._running = True
        self._stop_event.clear()

        try:
            await self._retry_with_backoff(self._connect, "Connection")
            self._connected = True
            self._task = asyncio.create_task(self._run_loop())
        except (OSError, ConnectionError, RuntimeError) as e:
            self._running = False
            logger.error(f"Failed to start CDC handler {self.connector_id}: {e}")
            raise

    async def _run_loop(self) -> None:
        """Run the main listen loop with reconnection support."""
        while self._running and not self._stop_event.is_set():
            try:
                await self._listen_loop()
            except asyncio.CancelledError:
                logger.info(f"CDC handler {self.connector_id} cancelled")
                break
            except (OSError, ConnectionError) as e:
                if self._running:
                    logger.warning(f"CDC connection lost: {e}. Reconnecting...")
                    self._connected = False
                    try:
                        await self._retry_with_backoff(self._connect, "Reconnection")
                        self._connected = True
                    except (OSError, ConnectionError) as e2:
                        logger.error(f"Reconnection failed: {e2}")
                        break

    async def stop(self) -> None:
        """Stop the CDC handler gracefully."""
        if not self._running:
            return

        logger.info(f"Stopping CDC handler: {self.connector_id}")
        self._running = False
        self._stop_event.set()

        if self._task:
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._task = None

        await self._disconnect()
        self._connected = False


class PostgresCDCHandler(BaseCDCHandler):
    """
    PostgreSQL CDC handler using LISTEN/NOTIFY.

    Listens to PostgreSQL notification channels for real-time
    change events. Requires:
    - asyncpg library
    - Database triggers to emit NOTIFY on changes

    Example trigger:
        CREATE OR REPLACE FUNCTION notify_changes()
        RETURNS TRIGGER AS $$
        BEGIN
            PERFORM pg_notify(
                'table_changes',
                json_build_object(
                    'operation', TG_OP,
                    'table', TG_TABLE_NAME,
                    'schema', TG_TABLE_SCHEMA,
                    'new_data', row_to_json(NEW),
                    'old_data', row_to_json(OLD)
                )::text
            );
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

    Usage:
        handler = PostgresCDCHandler(
            host="localhost",
            database="mydb",
            channels=["table_changes"],
            event_handler=KnowledgeMoundHandler(),
        )
        await handler.start()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        schema: str = "public",
        channels: list[str] | None = None,
        username: str | None = None,
        password: str | None = None,
        ssl: bool = False,
        **kwargs: Any,
    ):
        connector_id = kwargs.pop("connector_id", f"pg_cdc_{host}_{database}")
        super().__init__(
            connector_id=connector_id,
            source_type=CDCSourceType.POSTGRESQL,
            **kwargs,
        )

        self.host = host
        self.port = port
        self.database = database
        self.schema = schema
        self.channels = channels or ["table_changes"]
        self.username = username
        self.password = password
        self.ssl = ssl

        self._conn: Any = None
        self._notification_queue: asyncio.Queue = asyncio.Queue()

    async def _connect(self) -> None:
        """Establish asyncpg connection."""
        try:
            import asyncpg
        except ImportError:
            logger.warning("asyncpg not installed. PostgreSQL CDC requires: pip install asyncpg")
            raise ImportError("asyncpg is required for PostgreSQL CDC")

        # Build connection parameters
        conn_params: dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
        }

        if self.username:
            conn_params["user"] = self.username
        if self.password:
            conn_params["password"] = self.password
        if self.ssl:
            conn_params["ssl"] = "require"

        self._conn = await asyncpg.connect(**conn_params)

        # Set up notification callback
        async def notification_handler(
            connection: Any, pid: int, channel: str, payload: str
        ) -> None:
            await self._notification_queue.put((channel, payload))

        # Subscribe to channels
        for channel in self.channels:
            await self._conn.add_listener(channel, notification_handler)
            logger.info(f"[PostgresCDC] Listening on channel: {channel}")

    async def _disconnect(self) -> None:
        """Close asyncpg connection."""
        if self._conn:
            for channel in self.channels:
                try:
                    await self._conn.remove_listener(channel, lambda *_: None)
                except (OSError, RuntimeError):
                    pass
            await self._conn.close()
            self._conn = None

    async def _listen_loop(self) -> None:
        """Process notifications from the queue."""
        while self._running and not self._stop_event.is_set():
            try:
                # Use timeout to allow checking stop event periodically
                channel, payload = await asyncio.wait_for(
                    self._notification_queue.get(),
                    timeout=1.0,
                )

                event = ChangeEvent.from_postgres_notify(
                    payload=payload,
                    channel=channel,
                    connector_id=self.connector_id,
                    database=self.database,
                    schema=self.schema,
                )

                await self._process_event(event)

            except asyncio.TimeoutError:
                # Check for stop event
                continue
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"[PostgresCDC] Failed to parse notification: {e}")

    async def execute_query(self, query: str, *args: Any) -> list[Any]:
        """Execute a query on the connection (for setup/maintenance)."""
        if not self._conn:
            raise RuntimeError("Not connected to PostgreSQL")
        return await self._conn.fetch(query, *args)


class MySQLCDCHandler(BaseCDCHandler):
    """
    MySQL CDC handler using binary log (binlog) replication.

    Connects to MySQL as a replication slave and receives
    row-level change events. Requires:
    - mysql-replication library
    - Binary logging enabled on MySQL server
    - Replication permissions for the user

    MySQL server configuration:
        [mysqld]
        server-id = 1
        log_bin = mysql-bin
        binlog_format = ROW
        binlog_row_image = FULL

    User permissions:
        GRANT REPLICATION SLAVE, REPLICATION CLIENT ON *.* TO 'user'@'%';

    Usage:
        handler = MySQLCDCHandler(
            host="localhost",
            database="mydb",
            username="replicator",
            password="secret",
            event_handler=KnowledgeMoundHandler(),
        )
        await handler.start()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        database: str | None = None,
        tables: list[str] | None = None,
        username: str | None = None,
        password: str | None = None,
        server_id: int = 100,
        blocking: bool = True,
        resume_stream: bool = True,
        **kwargs: Any,
    ):
        connector_id = kwargs.pop("connector_id", f"mysql_cdc_{host}_{database or 'all'}")
        super().__init__(
            connector_id=connector_id,
            source_type=CDCSourceType.MYSQL,
            **kwargs,
        )

        self.host = host
        self.port = port
        self.database = database
        self.tables = tables
        self.username = username
        self.password = password
        self.server_id = server_id
        self.blocking = blocking
        self.resume_stream = resume_stream

        self._stream: Any = None
        self._stream_thread: Any = None

    async def _connect(self) -> None:
        """Create binlog stream reader."""
        try:
            from pymysqlreplication import BinLogStreamReader
            from pymysqlreplication.row_event import (
                DeleteRowsEvent,
                UpdateRowsEvent,
                WriteRowsEvent,
            )
        except ImportError:
            logger.warning(
                "mysql-replication not installed. MySQL CDC requires: pip install mysql-replication"
            )
            raise ImportError("mysql-replication is required for MySQL CDC")

        mysql_settings = {
            "host": self.host,
            "port": self.port,
            "user": self.username or "root",
            "passwd": self.password or "",
        }

        # Get resume position if available
        log_file = None
        log_pos = None
        resume_token = self.get_resume_token()
        if resume_token and self.resume_stream:
            try:
                token_data = json.loads(resume_token)
                log_file = token_data.get("log_file")
                log_pos = token_data.get("log_pos")
                logger.info(f"[MySQLCDC] Resuming from {log_file}:{log_pos}")
            except json.JSONDecodeError:
                pass

        stream_kwargs: dict[str, Any] = {
            "connection_settings": mysql_settings,
            "server_id": self.server_id,
            "blocking": self.blocking,
            "only_events": [WriteRowsEvent, UpdateRowsEvent, DeleteRowsEvent],
            "resume_stream": self.resume_stream,
        }

        if self.database:
            stream_kwargs["only_schemas"] = [self.database]
        if self.tables:
            stream_kwargs["only_tables"] = self.tables
        if log_file:
            stream_kwargs["log_file"] = log_file
        if log_pos:
            stream_kwargs["log_pos"] = log_pos

        # Create stream in a thread-safe way
        self._stream = BinLogStreamReader(**stream_kwargs)
        logger.info(f"[MySQLCDC] Connected to binlog stream on {self.host}")

    async def _disconnect(self) -> None:
        """Close binlog stream."""
        if self._stream:
            self._stream.close()
            self._stream = None

    async def _listen_loop(self) -> None:
        """Process binlog events in an async-friendly way."""
        try:
            from pymysqlreplication.row_event import (
                DeleteRowsEvent,
                UpdateRowsEvent,
                WriteRowsEvent,
            )
        except ImportError:
            return

        loop = asyncio.get_event_loop()

        def get_next_event() -> Any:
            """Get next event from binlog stream (blocking)."""
            if self._stream is None:
                return None
            for binlog_event in self._stream:
                return binlog_event
            return None

        while self._running and not self._stop_event.is_set():
            try:
                # Run blocking binlog read in executor
                binlog_event = await asyncio.wait_for(
                    loop.run_in_executor(None, get_next_event),
                    timeout=1.0,
                )

                if binlog_event is None:
                    continue

                # Map event type to operation
                if isinstance(binlog_event, WriteRowsEvent):
                    operation = ChangeOperation.INSERT
                elif isinstance(binlog_event, UpdateRowsEvent):
                    operation = ChangeOperation.UPDATE
                elif isinstance(binlog_event, DeleteRowsEvent):
                    operation = ChangeOperation.DELETE
                else:
                    continue

                # Process each row in the event
                for row in binlog_event.rows:
                    if operation == ChangeOperation.UPDATE:
                        data = row.get("after_values", {})
                        old_data = row.get("before_values", {})
                        fields_changed = [k for k in data.keys() if data.get(k) != old_data.get(k)]
                    elif operation == ChangeOperation.DELETE:
                        data = None
                        old_data = row.get("values", {})
                        fields_changed = []
                    else:  # INSERT
                        data = row.get("values", {})
                        old_data = None
                        fields_changed = []

                    # Extract primary key
                    pk_value = None
                    if data:
                        pk_value = data.get("id") or data.get("_id")
                    elif old_data:
                        pk_value = old_data.get("id") or old_data.get("_id")

                    # Build resume token from binlog position
                    resume_token = json.dumps(
                        {
                            "log_file": self._stream.log_file if self._stream else None,
                            "log_pos": self._stream.log_pos if self._stream else None,
                        }
                    )

                    event = ChangeEvent(
                        id="",
                        source_type=CDCSourceType.MYSQL,
                        connector_id=self.connector_id,
                        operation=operation,
                        timestamp=datetime.now(timezone.utc),
                        database=binlog_event.schema,
                        table=binlog_event.table,
                        primary_key={"id": pk_value} if pk_value else None,
                        data=data,
                        old_data=old_data,
                        fields_changed=fields_changed,
                        resume_token=resume_token,
                    )

                    await self._process_event(event)

            except asyncio.TimeoutError:
                # Allow checking stop event
                continue
            except (OSError, ConnectionError) as e:
                logger.error(f"[MySQLCDC] Binlog error: {e}")
                raise


class MongoDBCDCHandler(BaseCDCHandler):
    """
    MongoDB CDC handler using change streams.

    Watches MongoDB collections for changes using the
    change streams API. Requires:
    - motor (async MongoDB driver) library
    - MongoDB replica set or sharded cluster

    Usage:
        handler = MongoDBCDCHandler(
            host="localhost",
            database="mydb",
            collections=["users", "orders"],
            event_handler=KnowledgeMoundHandler(),
        )
        await handler.start()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 27017,
        database: str = "test",
        collections: list[str] | None = None,
        connection_string: str | None = None,
        username: str | None = None,
        password: str | None = None,
        watch_database: bool = True,
        full_document: str = "updateLookup",
        full_document_before_change: str | None = None,
        **kwargs: Any,
    ):
        connector_id = kwargs.pop("connector_id", f"mongo_cdc_{host}_{database}")
        super().__init__(
            connector_id=connector_id,
            source_type=CDCSourceType.MONGODB,
            **kwargs,
        )

        self.host = host
        self.port = port
        self.database_name = database
        self.collections = collections
        self.connection_string = connection_string
        self.username = username
        self.password = password
        self.watch_database = watch_database
        self.full_document = full_document
        self.full_document_before_change = full_document_before_change

        self._client: Any = None
        self._db: Any = None

    async def _connect(self) -> None:
        """Connect to MongoDB."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError:
            logger.warning("motor not installed. MongoDB CDC requires: pip install motor")
            raise ImportError("motor is required for MongoDB CDC")

        # Build connection string
        if self.connection_string:
            conn_str = self.connection_string
        elif self.username and self.password:
            conn_str = (
                f"mongodb://{self.username}:{self.password}@"
                f"{self.host}:{self.port}/{self.database_name}"
            )
        else:
            conn_str = f"mongodb://{self.host}:{self.port}/{self.database_name}"

        self._client = AsyncIOMotorClient(conn_str)
        self._db = self._client[self.database_name]

        # Verify connection
        await self._client.admin.command("ping")
        logger.info(f"[MongoDBCDC] Connected to {self.host}:{self.port}/{self.database_name}")

    async def _disconnect(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    async def _listen_loop(self) -> None:
        """Watch for changes using change streams."""
        # Build pipeline for filtering operations
        pipeline = [
            {"$match": {"operationType": {"$in": ["insert", "update", "replace", "delete"]}}}
        ]

        # Filter by collections if specified
        if self.collections and not self.watch_database:
            pipeline[0]["$match"]["ns.coll"] = {"$in": self.collections}

        # Build watch options
        watch_options: dict[str, Any] = {
            "pipeline": pipeline,
            "full_document": self.full_document,
        }

        # Add before-change image if supported (MongoDB 6.0+)
        if self.full_document_before_change:
            watch_options["full_document_before_change"] = self.full_document_before_change

        # Get resume token
        resume_token = self.get_resume_token()
        if resume_token:
            try:
                watch_options["resume_after"] = json.loads(resume_token)
                logger.info("[MongoDBCDC] Resuming from saved token")
            except json.JSONDecodeError:
                logger.warning("[MongoDBCDC] Invalid resume token, starting fresh")

        # Determine what to watch
        if self.collections and len(self.collections) == 1:
            # Watch a single collection
            target = self._db[self.collections[0]]
        elif self.watch_database:
            # Watch entire database
            target = self._db
        else:
            # Watch all collections explicitly listed
            target = self._db

        try:
            async with target.watch(**watch_options) as stream:
                logger.info("[MongoDBCDC] Change stream started")
                async for change in stream:
                    if self._stop_event.is_set():
                        break

                    # Filter by collection if watching database
                    ns = change.get("ns", {})
                    coll = ns.get("coll", "")
                    if self.collections and coll not in self.collections:
                        continue

                    event = ChangeEvent.from_mongodb_change(
                        change=change,
                        connector_id=self.connector_id,
                    )

                    await self._process_event(event)

        except asyncio.CancelledError:
            raise
        except (OSError, ConnectionError) as e:
            logger.error(f"[MongoDBCDC] Change stream error: {e}")
            raise


# =============================================================================
# Convenience Factory Functions
# =============================================================================


def create_postgres_cdc(
    host: str = "localhost",
    port: int = 5432,
    database: str = "postgres",
    channels: list[str] | None = None,
    username: str | None = None,
    password: str | None = None,
    event_handler: ChangeEventHandler | None = None,
    config: CDCConfig | None = None,
) -> PostgresCDCHandler:
    """
    Create a PostgreSQL CDC handler.

    Args:
        host: PostgreSQL host
        port: PostgreSQL port
        database: Database name
        channels: NOTIFY channels to listen on
        username: Database username
        password: Database password
        event_handler: Handler for change events
        config: CDC configuration

    Returns:
        Configured PostgresCDCHandler
    """
    return PostgresCDCHandler(
        host=host,
        port=port,
        database=database,
        channels=channels,
        username=username,
        password=password,
        event_handler=event_handler,
        config=config,
    )


def create_mysql_cdc(
    host: str = "localhost",
    port: int = 3306,
    database: str | None = None,
    tables: list[str] | None = None,
    username: str | None = None,
    password: str | None = None,
    server_id: int = 100,
    event_handler: ChangeEventHandler | None = None,
    config: CDCConfig | None = None,
) -> MySQLCDCHandler:
    """
    Create a MySQL CDC handler.

    Args:
        host: MySQL host
        port: MySQL port
        database: Database to watch (None = all)
        tables: Tables to watch (None = all)
        username: Database username (needs REPLICATION privileges)
        password: Database password
        server_id: Unique server ID for replication
        event_handler: Handler for change events
        config: CDC configuration

    Returns:
        Configured MySQLCDCHandler
    """
    return MySQLCDCHandler(
        host=host,
        port=port,
        database=database,
        tables=tables,
        username=username,
        password=password,
        server_id=server_id,
        event_handler=event_handler,
        config=config,
    )


def create_mongodb_cdc(
    host: str = "localhost",
    port: int = 27017,
    database: str = "test",
    collections: list[str] | None = None,
    connection_string: str | None = None,
    username: str | None = None,
    password: str | None = None,
    event_handler: ChangeEventHandler | None = None,
    config: CDCConfig | None = None,
) -> MongoDBCDCHandler:
    """
    Create a MongoDB CDC handler.

    Args:
        host: MongoDB host
        port: MongoDB port
        database: Database to watch
        collections: Collections to watch (None = all)
        connection_string: Full connection string (overrides host/port)
        username: Database username
        password: Database password
        event_handler: Handler for change events
        config: CDC configuration

    Returns:
        Configured MongoDBCDCHandler
    """
    return MongoDBCDCHandler(
        host=host,
        port=port,
        database=database,
        collections=collections,
        connection_string=connection_string,
        username=username,
        password=password,
        event_handler=event_handler,
        config=config,
    )
