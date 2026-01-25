"""
Tests for CDC (Change Data Capture) module.

Tests cover:
- ChangeEvent model and factory methods
- ChangeOperation and CDCSourceType enums
- ResumeToken and ResumeTokenStore persistence
- ChangeEventHandler implementations
- CDCStreamManager coordination
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# Enum Tests
# =============================================================================


class TestChangeOperation:
    """Tests for ChangeOperation enum."""

    def test_operation_values(self):
        """ChangeOperation enum has expected values."""
        from aragora.connectors.enterprise.database.cdc import ChangeOperation

        assert ChangeOperation.INSERT.value == "insert"
        assert ChangeOperation.UPDATE.value == "update"
        assert ChangeOperation.DELETE.value == "delete"
        assert ChangeOperation.REPLACE.value == "replace"
        assert ChangeOperation.UPSERT.value == "upsert"
        assert ChangeOperation.TRUNCATE.value == "truncate"
        assert ChangeOperation.SCHEMA_CHANGE.value == "schema_change"


class TestCDCSourceType:
    """Tests for CDCSourceType enum."""

    def test_source_type_values(self):
        """CDCSourceType enum has expected values."""
        from aragora.connectors.enterprise.database.cdc import CDCSourceType

        assert CDCSourceType.POSTGRESQL.value == "postgresql"
        assert CDCSourceType.MONGODB.value == "mongodb"
        assert CDCSourceType.SNOWFLAKE.value == "snowflake"
        assert CDCSourceType.MYSQL.value == "mysql"
        assert CDCSourceType.SQLSERVER.value == "sqlserver"


# =============================================================================
# ChangeEvent Tests
# =============================================================================


class TestChangeEvent:
    """Tests for ChangeEvent dataclass."""

    def test_event_creation(self):
        """Create a change event with required fields."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        event = ChangeEvent(
            id="test-event-1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="postgres_localhost_mydb_public",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="mydb",
            table="users",
        )

        assert event.id == "test-event-1"
        assert event.source_type == CDCSourceType.POSTGRESQL
        assert event.operation == ChangeOperation.INSERT
        assert event.database == "mydb"
        assert event.table == "users"

    def test_event_auto_generates_id(self):
        """Event generates ID if not provided."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        event = ChangeEvent(
            id="",
            source_type=CDCSourceType.MONGODB,
            connector_id="mongo_localhost_test",
            operation=ChangeOperation.UPDATE,
            timestamp=datetime.now(timezone.utc),
            database="test",
            table="products",
        )

        assert event.id  # Should have auto-generated ID
        assert len(event.id) == 16  # SHA256 truncated to 16 chars

    def test_is_data_change(self):
        """Check is_data_change property."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        now = datetime.now(timezone.utc)

        data_event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=now,
            database="db",
            table="t",
        )
        assert data_event.is_data_change is True

        schema_event = ChangeEvent(
            id="e2",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.SCHEMA_CHANGE,
            timestamp=now,
            database="db",
            table="t",
        )
        assert schema_event.is_data_change is False

    def test_qualified_table_with_schema(self):
        """Get qualified table name with schema."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.UPDATE,
            timestamp=datetime.now(timezone.utc),
            database="db",
            schema="public",
            table="users",
        )

        assert event.qualified_table == "public.users"

    def test_qualified_table_without_schema(self):
        """Get qualified table name without schema."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.MONGODB,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="db",
            table="users",
        )

        assert event.qualified_table == "users"

    def test_to_dict_serialization(self):
        """Serialize event to dictionary."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        now = datetime.now(timezone.utc)
        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=now,
            database="db",
            schema="public",
            table="users",
            data={"name": "John"},
            primary_key={"id": 1},
        )

        result = event.to_dict()

        assert result["id"] == "e1"
        assert result["source_type"] == "postgresql"
        assert result["operation"] == "insert"
        assert result["timestamp"] == now.isoformat()
        assert result["data"] == {"name": "John"}
        assert result["primary_key"] == {"id": 1}

    def test_from_dict_deserialization(self):
        """Deserialize event from dictionary."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        data = {
            "id": "e1",
            "source_type": "postgresql",
            "connector_id": "conn1",
            "operation": "update",
            "timestamp": "2025-01-15T12:00:00+00:00",
            "database": "db",
            "schema": "public",
            "table": "users",
            "data": {"name": "Jane"},
        }

        event = ChangeEvent.from_dict(data)

        assert event.id == "e1"
        assert event.source_type == CDCSourceType.POSTGRESQL
        assert event.operation == ChangeOperation.UPDATE
        assert event.data == {"name": "Jane"}


class TestChangeEventPostgresNotify:
    """Tests for PostgreSQL NOTIFY parsing."""

    def test_from_postgres_notify_json_payload(self):
        """Parse JSON payload from PostgreSQL NOTIFY."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        payload = json.dumps(
            {
                "operation": "INSERT",
                "table": "users",
                "new_data": {"id": 1, "name": "John"},
                "primary_key": {"id": 1},
            }
        )

        event = ChangeEvent.from_postgres_notify(
            payload=payload,
            channel="users_changes",
            connector_id="postgres_localhost_mydb_public",
            database="mydb",
            schema="public",
        )

        assert event.source_type == CDCSourceType.POSTGRESQL
        assert event.operation == ChangeOperation.INSERT
        assert event.database == "mydb"
        assert event.schema == "public"
        assert event.table == "users"
        assert event.data == {"id": 1, "name": "John"}
        assert event.primary_key == {"id": 1}

    def test_from_postgres_notify_update_operation(self):
        """Parse UPDATE operation from PostgreSQL."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
        )

        payload = json.dumps(
            {
                "operation": "UPDATE",
                "table": "products",
                "old_data": {"price": 99},
                "new_data": {"price": 149},
                "changed_columns": ["price"],
            }
        )

        event = ChangeEvent.from_postgres_notify(
            payload=payload,
            channel="products_changes",
            connector_id="conn1",
            database="db",
            schema="public",
        )

        assert event.operation == ChangeOperation.UPDATE
        assert event.old_data == {"price": 99}
        assert event.data == {"price": 149}
        assert event.fields_changed == ["price"]

    def test_from_postgres_notify_empty_payload(self):
        """Handle empty payload from PostgreSQL."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
        )

        event = ChangeEvent.from_postgres_notify(
            payload="",
            channel="my_channel",
            connector_id="conn1",
            database="db",
            schema="public",
        )

        # Should default to UPDATE and use channel as table
        assert event.operation == ChangeOperation.UPDATE
        assert event.table == "my_channel"

    def test_from_postgres_notify_invalid_json(self):
        """Handle invalid JSON payload."""
        from aragora.connectors.enterprise.database.cdc import ChangeEvent

        event = ChangeEvent.from_postgres_notify(
            payload="not valid json {",
            channel="test_channel",
            connector_id="conn1",
            database="db",
            schema="public",
        )

        # Should store raw payload in metadata
        assert event.metadata.get("raw_payload") == "not valid json {"


class TestChangeEventMongoDBChange:
    """Tests for MongoDB change stream parsing."""

    def test_from_mongodb_change_insert(self):
        """Parse insert from MongoDB change stream."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
        )

        change = {
            "operationType": "insert",
            "ns": {"db": "test", "coll": "users"},
            "documentKey": {"_id": "abc123"},
            "fullDocument": {"_id": "abc123", "name": "John", "email": "john@example.com"},
            "_id": {"_data": "resume_token_data"},
        }

        event = ChangeEvent.from_mongodb_change(
            change=change,
            connector_id="mongo_localhost_test",
        )

        assert event.source_type == CDCSourceType.MONGODB
        assert event.operation == ChangeOperation.INSERT
        assert event.database == "test"
        assert event.table == "users"
        assert event.document_id == "abc123"
        assert event.data == {"_id": "abc123", "name": "John", "email": "john@example.com"}
        assert event.resume_token is not None

    def test_from_mongodb_change_update(self):
        """Parse update from MongoDB change stream."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
        )

        change = {
            "operationType": "update",
            "ns": {"db": "test", "coll": "products"},
            "documentKey": {"_id": "prod456"},
            "updateDescription": {
                "updatedFields": {"price": 199, "stock": 50},
                "removedFields": ["discount"],
            },
        }

        event = ChangeEvent.from_mongodb_change(
            change=change,
            connector_id="conn1",
        )

        assert event.operation == ChangeOperation.UPDATE
        assert event.document_id == "prod456"
        assert set(event.fields_changed) == {"price", "stock", "discount"}

    def test_from_mongodb_change_delete(self):
        """Parse delete from MongoDB change stream."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
        )

        change = {
            "operationType": "delete",
            "ns": {"db": "test", "coll": "sessions"},
            "documentKey": {"_id": "sess789"},
        }

        event = ChangeEvent.from_mongodb_change(
            change=change,
            connector_id="conn1",
        )

        assert event.operation == ChangeOperation.DELETE
        assert event.document_id == "sess789"

    def test_from_mongodb_change_replace(self):
        """Parse replace from MongoDB change stream."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
        )

        change = {
            "operationType": "replace",
            "ns": {"db": "test", "coll": "configs"},
            "documentKey": {"_id": "config1"},
            "fullDocument": {"_id": "config1", "settings": {"theme": "dark"}},
            "fullDocumentBeforeChange": {"_id": "config1", "settings": {"theme": "light"}},
        }

        event = ChangeEvent.from_mongodb_change(
            change=change,
            connector_id="conn1",
        )

        assert event.operation == ChangeOperation.REPLACE
        assert event.data == {"_id": "config1", "settings": {"theme": "dark"}}
        assert event.old_data == {"_id": "config1", "settings": {"theme": "light"}}


# =============================================================================
# ResumeToken Tests
# =============================================================================


class TestResumeToken:
    """Tests for ResumeToken dataclass."""

    def test_token_creation(self):
        """Create a resume token."""
        from aragora.connectors.enterprise.database.cdc import (
            ResumeToken,
            CDCSourceType,
        )

        now = datetime.now(timezone.utc)
        token = ResumeToken(
            connector_id="mongo_localhost_test",
            source_type=CDCSourceType.MONGODB,
            token='{"_data": "resume_token"}',
            timestamp=now,
            sequence_number=100,
        )

        assert token.connector_id == "mongo_localhost_test"
        assert token.source_type == CDCSourceType.MONGODB
        assert token.token == '{"_data": "resume_token"}'
        assert token.timestamp == now
        assert token.sequence_number == 100

    def test_token_to_dict(self):
        """Serialize token to dictionary."""
        from aragora.connectors.enterprise.database.cdc import (
            ResumeToken,
            CDCSourceType,
        )

        now = datetime.now(timezone.utc)
        token = ResumeToken(
            connector_id="conn1",
            source_type=CDCSourceType.POSTGRESQL,
            token="LSN:0/16B3740",
            timestamp=now,
        )

        result = token.to_dict()

        assert result["connector_id"] == "conn1"
        assert result["source_type"] == "postgresql"
        assert result["token"] == "LSN:0/16B3740"
        assert result["timestamp"] == now.isoformat()

    def test_token_from_dict(self):
        """Deserialize token from dictionary."""
        from aragora.connectors.enterprise.database.cdc import (
            ResumeToken,
            CDCSourceType,
        )

        data = {
            "connector_id": "conn1",
            "source_type": "mongodb",
            "token": '{"_data": "token"}',
            "timestamp": "2025-01-15T12:00:00+00:00",
            "sequence_number": 42,
        }

        token = ResumeToken.from_dict(data)

        assert token.connector_id == "conn1"
        assert token.source_type == CDCSourceType.MONGODB
        assert token.sequence_number == 42


class TestResumeTokenStore:
    """Tests for ResumeTokenStore persistence."""

    def test_store_and_get_token(self):
        """Store and retrieve a resume token."""
        from aragora.connectors.enterprise.database.cdc import (
            ResumeToken,
            ResumeTokenStore,
            CDCSourceType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            store = ResumeTokenStore(storage_path=storage_path)

            token = ResumeToken(
                connector_id="conn1",
                source_type=CDCSourceType.MONGODB,
                token='{"_data": "test_token"}',
                timestamp=datetime.now(timezone.utc),
            )

            store.save(token)
            retrieved = store.get("conn1")

            assert retrieved is not None
            assert retrieved.token == '{"_data": "test_token"}'

    def test_get_nonexistent_token(self):
        """Get returns None for nonexistent token."""
        from aragora.connectors.enterprise.database.cdc import ResumeTokenStore

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            store = ResumeTokenStore(storage_path=storage_path)

            result = store.get("nonexistent")

            assert result is None

    def test_delete_token(self):
        """Delete a resume token."""
        from aragora.connectors.enterprise.database.cdc import (
            ResumeToken,
            ResumeTokenStore,
            CDCSourceType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            store = ResumeTokenStore(storage_path=storage_path)

            token = ResumeToken(
                connector_id="conn1",
                source_type=CDCSourceType.POSTGRESQL,
                token="token1",
                timestamp=datetime.now(timezone.utc),
            )

            store.save(token)
            assert store.get("conn1") is not None

            store.delete("conn1")
            assert store.get("conn1") is None

    def test_clear_all_tokens(self):
        """Clear all resume tokens."""
        from aragora.connectors.enterprise.database.cdc import (
            ResumeToken,
            ResumeTokenStore,
            CDCSourceType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            store = ResumeTokenStore(storage_path=storage_path)

            now = datetime.now(timezone.utc)
            for i in range(3):
                store.save(
                    ResumeToken(
                        connector_id=f"conn{i}",
                        source_type=CDCSourceType.MONGODB,
                        token=f"token{i}",
                        timestamp=now,
                    )
                )

            store.clear_all()

            for i in range(3):
                assert store.get(f"conn{i}") is None

    def test_persistence_across_instances(self):
        """Tokens persist across store instances."""
        from aragora.connectors.enterprise.database.cdc import (
            ResumeToken,
            ResumeTokenStore,
            CDCSourceType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"

            # First instance saves token
            store1 = ResumeTokenStore(storage_path=storage_path)
            store1.save(
                ResumeToken(
                    connector_id="conn1",
                    source_type=CDCSourceType.MONGODB,
                    token="persistent_token",
                    timestamp=datetime.now(timezone.utc),
                )
            )

            # Second instance should load token
            store2 = ResumeTokenStore(storage_path=storage_path)
            retrieved = store2.get("conn1")

            assert retrieved is not None
            assert retrieved.token == "persistent_token"


# =============================================================================
# Handler Tests
# =============================================================================


class TestCallbackHandler:
    """Tests for CallbackHandler."""

    @pytest.mark.asyncio
    async def test_sync_callback(self):
        """Handle event with sync callback."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            CallbackHandler,
        )

        events_received = []

        def callback(event):
            events_received.append(event)
            return True

        handler = CallbackHandler(callback)

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="db",
            table="users",
        )

        result = await handler.handle(event)

        assert result is True
        assert len(events_received) == 1
        assert events_received[0].id == "e1"

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Handle event with async callback."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            CallbackHandler,
        )

        events_received = []

        async def async_callback(event):
            events_received.append(event)
            return True

        handler = CallbackHandler(async_callback)

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.MONGODB,
            connector_id="conn1",
            operation=ChangeOperation.UPDATE,
            timestamp=datetime.now(timezone.utc),
            database="db",
            table="products",
        )

        result = await handler.handle(event)

        assert result is True
        assert len(events_received) == 1


class TestCompositeHandler:
    """Tests for CompositeHandler."""

    @pytest.mark.asyncio
    async def test_delegates_to_all_handlers(self):
        """Composite handler calls all sub-handlers."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            CompositeHandler,
            CallbackHandler,
        )

        calls = {"h1": 0, "h2": 0}

        def callback1(e):
            calls["h1"] += 1
            return True

        def callback2(e):
            calls["h2"] += 1
            return True

        composite = CompositeHandler(
            [
                CallbackHandler(callback1),
                CallbackHandler(callback2),
            ]
        )

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="db",
            table="t",
        )

        result = await composite.handle(event)

        assert result is True
        assert calls["h1"] == 1
        assert calls["h2"] == 1

    @pytest.mark.asyncio
    async def test_returns_false_if_any_handler_fails(self):
        """Composite returns False if any handler fails."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            CompositeHandler,
            CallbackHandler,
        )

        composite = CompositeHandler(
            [
                CallbackHandler(lambda e: True),
                CallbackHandler(lambda e: False),  # This one fails
            ]
        )

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="db",
            table="t",
        )

        result = await composite.handle(event)

        assert result is False

    @pytest.mark.asyncio
    async def test_continues_after_handler_exception(self):
        """Composite continues even if a handler throws."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            CompositeHandler,
            CallbackHandler,
        )

        calls = {"h2": 0}

        def throwing_callback(e):
            raise RuntimeError("Handler error")

        def success_callback(e):
            calls["h2"] += 1
            return True

        composite = CompositeHandler(
            [
                CallbackHandler(throwing_callback),
                CallbackHandler(success_callback),
            ]
        )

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="db",
            table="t",
        )

        result = await composite.handle(event)

        # Should return False due to exception, but second handler still called
        assert result is False
        assert calls["h2"] == 1

    def test_add_handler(self):
        """Add handler to composite."""
        from aragora.connectors.enterprise.database.cdc import (
            CompositeHandler,
            CallbackHandler,
        )

        composite = CompositeHandler()
        assert len(composite.handlers) == 0

        composite.add_handler(CallbackHandler(lambda e: True))
        assert len(composite.handlers) == 1


# =============================================================================
# CDCStreamManager Tests
# =============================================================================


class TestCDCStreamManager:
    """Tests for CDCStreamManager."""

    def test_manager_creation(self):
        """Create a stream manager."""
        from aragora.connectors.enterprise.database.cdc import (
            CDCStreamManager,
            CDCSourceType,
        )

        manager = CDCStreamManager(
            connector_id="conn1",
            source_type=CDCSourceType.MONGODB,
        )

        assert manager.connector_id == "conn1"
        assert manager.source_type == CDCSourceType.MONGODB
        assert manager.is_running is False

    def test_start_and_stop(self):
        """Start and stop the stream manager."""
        from aragora.connectors.enterprise.database.cdc import (
            CDCStreamManager,
            CDCSourceType,
        )

        manager = CDCStreamManager(
            connector_id="conn1",
            source_type=CDCSourceType.POSTGRESQL,
        )

        manager.start()
        assert manager.is_running is True

        manager.stop()
        assert manager.is_running is False

    def test_stats(self):
        """Get manager statistics."""
        from aragora.connectors.enterprise.database.cdc import (
            CDCStreamManager,
            CDCSourceType,
        )

        manager = CDCStreamManager(
            connector_id="conn1",
            source_type=CDCSourceType.MONGODB,
        )

        stats = manager.stats

        assert stats["connector_id"] == "conn1"
        assert stats["source_type"] == "mongodb"
        assert stats["running"] is False
        assert stats["events_processed"] == 0
        assert stats["last_event_time"] is None

    @pytest.mark.asyncio
    async def test_process_event(self):
        """Process an event through the manager."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            CDCStreamManager,
            CallbackHandler,
        )

        events_received = []

        manager = CDCStreamManager(
            connector_id="conn1",
            source_type=CDCSourceType.POSTGRESQL,
            handler=CallbackHandler(lambda e: events_received.append(e) or True),
        )

        event = ChangeEvent(
            id="e1",
            source_type=CDCSourceType.POSTGRESQL,
            connector_id="conn1",
            operation=ChangeOperation.INSERT,
            timestamp=datetime.now(timezone.utc),
            database="db",
            table="t",
        )

        result = await manager.process_event(event)

        assert result is True
        assert len(events_received) == 1
        assert manager.stats["events_processed"] == 1

    @pytest.mark.asyncio
    async def test_saves_resume_token(self):
        """Manager saves resume token from event."""
        from aragora.connectors.enterprise.database.cdc import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            CDCStreamManager,
            CallbackHandler,
            ResumeTokenStore,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            token_store = ResumeTokenStore(storage_path=storage_path)

            manager = CDCStreamManager(
                connector_id="conn1",
                source_type=CDCSourceType.MONGODB,
                handler=CallbackHandler(lambda e: True),
                token_store=token_store,
            )

            event = ChangeEvent(
                id="e1",
                source_type=CDCSourceType.MONGODB,
                connector_id="conn1",
                operation=ChangeOperation.INSERT,
                timestamp=datetime.now(timezone.utc),
                database="db",
                table="t",
                resume_token='{"_data": "my_resume_token"}',
            )

            await manager.process_event(event)

            saved_token = manager.get_resume_token()
            assert saved_token == '{"_data": "my_resume_token"}'

    def test_reset_clears_state(self):
        """Reset clears manager state."""
        from aragora.connectors.enterprise.database.cdc import (
            CDCStreamManager,
            CDCSourceType,
            ResumeTokenStore,
            ResumeToken,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "tokens.json"
            token_store = ResumeTokenStore(storage_path=storage_path)

            # Save a token
            token_store.save(
                ResumeToken(
                    connector_id="conn1",
                    source_type=CDCSourceType.MONGODB,
                    token="old_token",
                    timestamp=datetime.now(timezone.utc),
                )
            )

            manager = CDCStreamManager(
                connector_id="conn1",
                source_type=CDCSourceType.MONGODB,
                token_store=token_store,
            )

            manager._events_processed = 100
            manager._last_event_time = datetime.now(timezone.utc)

            manager.reset()

            assert manager._events_processed == 0
            assert manager._last_event_time is None
            assert manager.get_resume_token() is None


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports from __init__.py."""

    def test_import_from_database_module(self):
        """Import CDC components from database module."""
        from aragora.connectors.enterprise.database import (
            ChangeEvent,
            ChangeOperation,
            CDCSourceType,
            ResumeToken,
            ResumeTokenStore,
            ChangeEventHandler,
            KnowledgeMoundHandler,
            CallbackHandler,
            CompositeHandler,
            CDCStreamManager,
        )

        # Verify imports work
        assert ChangeEvent is not None
        assert ChangeOperation is not None
        assert CDCSourceType is not None
        assert ResumeToken is not None
        assert ResumeTokenStore is not None
        assert ChangeEventHandler is not None
        assert KnowledgeMoundHandler is not None
        assert CallbackHandler is not None
        assert CompositeHandler is not None
        assert CDCStreamManager is not None
