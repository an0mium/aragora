"""
CDC Event Handlers and Stream Manager.

Provides abstract and concrete handler classes for processing
Change Data Capture events, including Knowledge Mound integration.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TYPE_CHECKING, cast
from collections.abc import Callable

from .cdc_models import (
    CDCSourceType,
    ChangeEvent,
    ChangeOperation,
    ResumeToken,
    ResumeTokenStore,
)

if TYPE_CHECKING:
    from aragora.knowledge.mound import KnowledgeMound

logger = logging.getLogger(__name__)


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
        self._mound: KnowledgeMound | None = None

    async def _get_mound(self) -> KnowledgeMound:
        """Get or create Knowledge Mound instance."""
        if self._mound is None:
            from aragora.knowledge.mound import KnowledgeMound

            # KnowledgeMound uses factory pattern - cast to satisfy type checker
            self._mound = cast("KnowledgeMound", KnowledgeMound(workspace_id=self.workspace_id))  # type: ignore[abstract, redundant-cast]
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

    async def _handle_upsert(self, mound: KnowledgeMound, event: ChangeEvent) -> None:
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

    async def _handle_delete(self, mound: KnowledgeMound, event: ChangeEvent) -> None:
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

    def __init__(self, handlers: list[ChangeEventHandler] | None = None):
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
