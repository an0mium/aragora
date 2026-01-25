"""
Integration tests for CDC (Change Data Capture) to KnowledgeMound flow.

Tests verify that database change events are correctly:
1. Processed by KnowledgeMoundHandler
2. Stored in KnowledgeMound
3. Searchable and retrievable
4. Error-handled gracefully
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.database.cdc import (
    CDCSourceType,
    ChangeEvent,
    ChangeOperation,
    KnowledgeMoundHandler,
)


# =============================================================================
# Fixtures
# =============================================================================


_UNSET = object()  # Sentinel for distinguishing None from unset


@pytest.fixture
def cdc_event_factory():
    """Factory for creating CDC events with various configurations."""

    def create(
        source_type: str = "postgresql",
        operation: str = "insert",
        table: str = "products",
        data: Any = _UNSET,
        primary_key: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        resume_token: Optional[str] = None,
    ) -> ChangeEvent:
        # Use sentinel to distinguish None/empty from "use default"
        event_data = {"name": "Test Product", "price": 99.99} if data is _UNSET else data
        return ChangeEvent(
            id="",
            source_type=CDCSourceType(source_type),
            connector_id=f"{source_type}_localhost_test",
            operation=ChangeOperation(operation),
            timestamp=datetime.now(timezone.utc),
            database="test_db",
            table=table,
            data=event_data,
            primary_key=primary_key or {"id": 1},
            document_id=document_id,
            resume_token=resume_token,
        )

    return create


@pytest.fixture
def mock_knowledge_mound():
    """Create mock KnowledgeMound that tracks store calls."""
    mound = MagicMock()
    mound.workspace_id = "test_workspace"
    mound._nodes: Dict[str, Dict[str, Any]] = {}
    mound._node_counter = 0

    async def mock_store(request):
        mound._node_counter += 1
        node_id = f"node_{mound._node_counter:03d}"
        mound._nodes[node_id] = {
            "id": node_id,
            "content": request.content,
            "topics": request.topics,
            "metadata": request.metadata,
            "confidence": request.confidence,
        }
        result = MagicMock()
        result.node_id = node_id
        result.success = True
        return result

    async def mock_query(query: str, **kwargs):
        # Simple substring matching
        return [n for n in mound._nodes.values() if query.lower() in n["content"].lower()]

    mound.store = AsyncMock(side_effect=mock_store)
    mound.query = AsyncMock(side_effect=mock_query)
    return mound


@pytest.fixture
def postgres_notify_payloads():
    """Sample PostgreSQL NOTIFY payloads."""
    return {
        "insert": json.dumps(
            {
                "operation": "INSERT",
                "table": "products",
                "new_data": {"id": 1, "name": "Widget", "price": 99.99},
                "primary_key": {"id": 1},
            }
        ),
        "update": json.dumps(
            {
                "operation": "UPDATE",
                "table": "products",
                "old_data": {"price": 99.99},
                "new_data": {"price": 149.99},
                "changed_columns": ["price"],
                "primary_key": {"id": 1},
            }
        ),
        "delete": json.dumps(
            {
                "operation": "DELETE",
                "table": "products",
                "primary_key": {"id": 1},
            }
        ),
    }


@pytest.fixture
def mongodb_change_stream_events():
    """Sample MongoDB change stream events."""
    return {
        "insert": {
            "operationType": "insert",
            "ns": {"db": "test_db", "coll": "products"},
            "documentKey": {"_id": "prod_001"},
            "fullDocument": {"_id": "prod_001", "name": "Widget", "price": 99.99},
            "_id": {"_data": "resume_token_001"},
        },
        "update": {
            "operationType": "update",
            "ns": {"db": "test_db", "coll": "products"},
            "documentKey": {"_id": "prod_001"},
            "updateDescription": {
                "updatedFields": {"price": 149.99},
                "removedFields": [],
            },
            "_id": {"_data": "resume_token_002"},
        },
        "delete": {
            "operationType": "delete",
            "ns": {"db": "test_db", "coll": "products"},
            "documentKey": {"_id": "prod_001"},
            "_id": {"_data": "resume_token_003"},
        },
    }


# =============================================================================
# Happy Path Tests
# =============================================================================


class TestCDCToKnowledgeMoundHappyPath:
    """Integration tests for CDC events flowing to Knowledge Mound."""

    @pytest.mark.asyncio
    async def test_insert_event_ingested_to_km(self, cdc_event_factory, mock_knowledge_mound):
        """INSERT event should be ingested into KnowledgeMound."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(
            operation="insert",
            table="products",
            data={"name": "Widget", "price": 99.99, "category": "gadgets"},
        )

        result = await handler.handle(event)

        assert result is True
        assert mock_knowledge_mound.store.await_count == 1

        # Verify stored content
        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "Widget" in stored["content"]
        assert "price: 99.99" in stored["content"]
        assert stored["topics"] == ["products"]
        assert stored["metadata"]["operation"] == "insert"
        assert stored["metadata"]["table"] == "products"

    @pytest.mark.asyncio
    async def test_update_event_ingested_to_km(self, cdc_event_factory, mock_knowledge_mound):
        """UPDATE event should be ingested into KnowledgeMound."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(
            operation="update",
            table="users",
            data={"username": "john_doe", "email": "john@example.com"},
        )

        result = await handler.handle(event)

        assert result is True
        assert mock_knowledge_mound.store.await_count == 1

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "john_doe" in stored["content"]
        assert stored["metadata"]["operation"] == "update"

    @pytest.mark.asyncio
    async def test_mongodb_insert_event_ingested_to_km(
        self, mock_knowledge_mound, mongodb_change_stream_events
    ):
        """MongoDB insert change stream event should be ingested."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = ChangeEvent(
            id="",
            source_type=CDCSourceType.MONGODB,
            connector_id="mongo_localhost_test",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="test_db",
            table="products",
            data=mongodb_change_stream_events["insert"]["fullDocument"],
            document_id="prod_001",
        )

        result = await handler.handle(event)

        assert result is True
        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert stored["metadata"]["source"] == "mongodb"

    @pytest.mark.asyncio
    async def test_delete_event_logged_not_stored(self, cdc_event_factory, mock_knowledge_mound):
        """DELETE events should be handled but not store new data."""
        handler = KnowledgeMoundHandler(workspace_id="test", delete_on_remove=True)
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(operation="delete", data=None)

        result = await handler.handle(event)

        assert result is True
        # Delete handler logs but doesn't store new data
        assert mock_knowledge_mound.store.await_count == 0

    @pytest.mark.asyncio
    async def test_batch_events_processed_sequentially(
        self, cdc_event_factory, mock_knowledge_mound
    ):
        """Multiple CDC events should be processed in order."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        events = [
            cdc_event_factory(operation="insert", data={"name": f"Product {i}", "price": i * 10})
            for i in range(5)
        ]

        results = []
        for event in events:
            result = await handler.handle(event)
            results.append(result)

        assert all(results)
        assert mock_knowledge_mound.store.await_count == 5
        assert len(mock_knowledge_mound._nodes) == 5

    @pytest.mark.asyncio
    async def test_high_confidence_data_stored_correctly(
        self, cdc_event_factory, mock_knowledge_mound
    ):
        """CDC data should be stored with appropriate confidence (0.8)."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(operation="insert")
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert stored["confidence"] == 0.8


# =============================================================================
# Data Indexing and Retrieval Tests
# =============================================================================


class TestCDCDataIndexingRetrieval:
    """Tests for verifying CDC data is properly indexed and retrievable."""

    @pytest.mark.asyncio
    async def test_metadata_preserved_on_ingestion(self, cdc_event_factory, mock_knowledge_mound):
        """CDC metadata (source, table, document_id) should be preserved."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(
            source_type="postgresql",
            table="orders",
            primary_key={"order_id": 12345},
            data={"customer": "Alice", "total": 250.00},
        )
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert stored["metadata"]["source"] == "postgresql"
        assert stored["metadata"]["table"] == "orders"
        assert stored["metadata"]["database"] == "test_db"
        assert "order_id" in str(stored["metadata"]["document_id"])

    @pytest.mark.asyncio
    async def test_table_name_becomes_topic(self, cdc_event_factory, mock_knowledge_mound):
        """Table/collection name should become knowledge topic."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(table="inventory")
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert stored["topics"] == ["inventory"]

    @pytest.mark.asyncio
    async def test_source_type_recorded(self, cdc_event_factory, mock_knowledge_mound):
        """Source type (postgresql/mongodb) should be recorded in metadata."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        # Test PostgreSQL
        pg_event = cdc_event_factory(source_type="postgresql")
        await handler.handle(pg_event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert stored["metadata"]["source"] == "postgresql"

    @pytest.mark.asyncio
    async def test_timestamp_preserved(self, cdc_event_factory, mock_knowledge_mound):
        """Event timestamp should be preserved in metadata."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(operation="insert")
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "timestamp" in stored["metadata"]
        # Should be ISO format string
        assert "T" in stored["metadata"]["timestamp"]


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCDCIngestionErrorHandling:
    """Tests for error handling during CDC to KM ingestion."""

    @pytest.mark.asyncio
    async def test_km_unavailable_returns_false(self, cdc_event_factory):
        """Handler should return False when KM is unavailable."""
        handler = KnowledgeMoundHandler(workspace_id="test")

        # Mock _get_mound to raise exception
        async def raise_error():
            raise RuntimeError("KM unavailable")

        handler._get_mound = raise_error

        event = cdc_event_factory(operation="insert")
        result = await handler.handle(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_store_failure_returns_false(self, cdc_event_factory, mock_knowledge_mound):
        """KM store failure should return False."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        # Make store raise an exception
        mock_knowledge_mound.store = AsyncMock(side_effect=RuntimeError("Store failed"))

        event = cdc_event_factory(operation="insert")
        result = await handler.handle(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_event_without_data_skipped(self, cdc_event_factory, mock_knowledge_mound):
        """Events without data should be skipped gracefully."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(operation="insert", data=None)
        result = await handler.handle(event)

        # Should succeed but not store anything
        assert result is True
        assert mock_knowledge_mound.store.await_count == 0

    @pytest.mark.asyncio
    async def test_empty_data_dict_skipped(self, cdc_event_factory, mock_knowledge_mound):
        """Events with empty data dict should be skipped."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(operation="insert", data={})
        result = await handler.handle(event)

        # Should succeed but not store (empty content)
        assert result is True
        assert mock_knowledge_mound.store.await_count == 0

    @pytest.mark.asyncio
    async def test_non_data_change_event_skipped(self, mock_knowledge_mound):
        """Non-data change events (schema changes) should be skipped."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = ChangeEvent(
            id="",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="test",
            operation=ChangeOperation.SCHEMA_CHANGE,
            timestamp=datetime.now(timezone.utc),
            database="test_db",
            table="users",
        )

        result = await handler.handle(event)

        assert result is True
        assert mock_knowledge_mound.store.await_count == 0


# =============================================================================
# Handler Configuration Tests
# =============================================================================


class TestKnowledgeMoundHandlerConfiguration:
    """Tests for KnowledgeMoundHandler configuration options."""

    def test_handler_default_configuration(self):
        """Handler has correct default configuration."""
        handler = KnowledgeMoundHandler()
        assert handler.workspace_id == "default"
        assert handler.auto_ingest is True
        assert handler.delete_on_remove is True

    def test_handler_custom_workspace(self):
        """Handler accepts custom workspace ID."""
        handler = KnowledgeMoundHandler(workspace_id="custom_workspace")
        assert handler.workspace_id == "custom_workspace"

    @pytest.mark.asyncio
    async def test_auto_ingest_disabled_skips_insert(self, cdc_event_factory, mock_knowledge_mound):
        """With auto_ingest=False, insert events should not be stored."""
        handler = KnowledgeMoundHandler(workspace_id="test", auto_ingest=False)
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(operation="insert")
        result = await handler.handle(event)

        assert result is True
        assert mock_knowledge_mound.store.await_count == 0

    @pytest.mark.asyncio
    async def test_delete_on_remove_disabled(self, cdc_event_factory, mock_knowledge_mound):
        """With delete_on_remove=False, delete events should be no-op."""
        handler = KnowledgeMoundHandler(workspace_id="test", delete_on_remove=False)
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(operation="delete", data=None)
        result = await handler.handle(event)

        assert result is True


# =============================================================================
# Content Formatting Tests
# =============================================================================


class TestCDCContentFormatting:
    """Tests for how CDC data is formatted as content."""

    @pytest.mark.asyncio
    async def test_data_fields_formatted_as_key_value(
        self, cdc_event_factory, mock_knowledge_mound
    ):
        """Data fields should be formatted as 'key: value' lines."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(data={"title": "Test Doc", "author": "John", "pages": 100})
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "title: Test Doc" in stored["content"]
        assert "author: John" in stored["content"]
        assert "pages: 100" in stored["content"]

    @pytest.mark.asyncio
    async def test_private_fields_excluded(self, cdc_event_factory, mock_knowledge_mound):
        """Fields starting with _ should be excluded from content."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(
            data={"name": "Public", "_internal": "Secret", "__meta": "Hidden"}
        )
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "name: Public" in stored["content"]
        assert "_internal" not in stored["content"]
        assert "__meta" not in stored["content"]

    @pytest.mark.asyncio
    async def test_nested_objects_serialized(self, cdc_event_factory, mock_knowledge_mound):
        """Nested objects should be JSON serialized."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(
            data={"name": "Product", "specs": {"weight": 100, "color": "red"}}
        )
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "specs:" in stored["content"]
        # Nested object should be JSON
        assert "weight" in stored["content"]

    @pytest.mark.asyncio
    async def test_datetime_formatted_as_iso(self, cdc_event_factory, mock_knowledge_mound):
        """Datetime values should be formatted as ISO strings."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        now = datetime.now(timezone.utc)
        event = cdc_event_factory(data={"created_at": now, "name": "Test"})
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "created_at:" in stored["content"]
        # Should contain ISO format
        assert "T" in stored["content"]  # ISO datetime has T separator

    @pytest.mark.asyncio
    async def test_none_values_excluded(self, cdc_event_factory, mock_knowledge_mound):
        """None values should be excluded from content."""
        handler = KnowledgeMoundHandler(workspace_id="test")
        handler._mound = mock_knowledge_mound

        event = cdc_event_factory(data={"name": "Test", "optional": None})
        await handler.handle(event)

        stored = list(mock_knowledge_mound._nodes.values())[0]
        assert "name: Test" in stored["content"]
        assert "optional" not in stored["content"]
