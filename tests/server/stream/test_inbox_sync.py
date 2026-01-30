"""
Tests for inbox_sync.py - WebSocket events for Gmail inbox synchronization.

Tests cover:
- InboxSyncEventType enum values
- InboxSyncEvent dataclass (serialization, timestamp generation)
- InboxSyncEmitter (subscription management, event emission, callbacks)
- Convenience functions (emit_sync_start, emit_sync_progress, etc.)
- Error handling for sync failures
- Concurrent inbox updates
- Dead connection cleanup
"""

import asyncio
import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.stream.inbox_sync import (
    InboxSyncEmitter,
    InboxSyncEvent,
    InboxSyncEventType,
    emit_new_priority_email,
    emit_sync_complete,
    emit_sync_error,
    emit_sync_progress,
    emit_sync_start,
    get_inbox_sync_emitter,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def emitter():
    """Create a fresh InboxSyncEmitter for each test."""
    return InboxSyncEmitter()


@pytest.fixture
def mock_ws():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def mock_ws_closed():
    """Create a mock WebSocket that raises on send (simulating closed connection)."""
    ws = MagicMock()
    ws.send = AsyncMock(side_effect=ConnectionError("Connection closed"))
    return ws


@pytest.fixture
def multiple_mock_ws():
    """Create multiple mock WebSocket connections."""
    return [
        MagicMock(send=AsyncMock()),
        MagicMock(send=AsyncMock()),
        MagicMock(send=AsyncMock()),
    ]


@pytest.fixture(autouse=True)
def reset_global_emitter():
    """Reset the global emitter before each test."""
    import aragora.server.stream.inbox_sync as inbox_sync_module

    inbox_sync_module._inbox_sync_emitter = None
    yield
    inbox_sync_module._inbox_sync_emitter = None


# ===========================================================================
# Test InboxSyncEventType
# ===========================================================================


class TestInboxSyncEventType:
    """Tests for InboxSyncEventType enum."""

    def test_sync_start_value(self):
        """SYNC_START has correct string value."""
        assert InboxSyncEventType.SYNC_START.value == "inbox_sync_start"

    def test_sync_progress_value(self):
        """SYNC_PROGRESS has correct string value."""
        assert InboxSyncEventType.SYNC_PROGRESS.value == "inbox_sync_progress"

    def test_sync_complete_value(self):
        """SYNC_COMPLETE has correct string value."""
        assert InboxSyncEventType.SYNC_COMPLETE.value == "inbox_sync_complete"

    def test_sync_error_value(self):
        """SYNC_ERROR has correct string value."""
        assert InboxSyncEventType.SYNC_ERROR.value == "inbox_sync_error"

    def test_new_priority_email_value(self):
        """NEW_PRIORITY_EMAIL has correct string value."""
        assert InboxSyncEventType.NEW_PRIORITY_EMAIL.value == "new_priority_email"

    def test_event_type_is_string_enum(self):
        """InboxSyncEventType is a string enum."""
        assert isinstance(InboxSyncEventType.SYNC_START, str)
        assert InboxSyncEventType.SYNC_START == "inbox_sync_start"


# ===========================================================================
# Test InboxSyncEvent
# ===========================================================================


class TestInboxSyncEvent:
    """Tests for InboxSyncEvent dataclass."""

    def test_event_creation_basic(self):
        """Event can be created with basic attributes."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={"total_messages": 100},
        )
        assert event.type == InboxSyncEventType.SYNC_START
        assert event.user_id == "user-123"
        assert event.data["total_messages"] == 100

    def test_event_auto_generates_timestamp(self):
        """Event auto-generates timestamp if not provided."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
        )
        assert event.timestamp is not None
        assert event.timestamp.endswith("Z")
        # Verify it's a valid ISO timestamp
        datetime.fromisoformat(event.timestamp.rstrip("Z"))

    def test_event_preserves_provided_timestamp(self):
        """Event preserves explicitly provided timestamp."""
        timestamp = "2024-01-15T10:30:00Z"
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
            timestamp=timestamp,
        )
        assert event.timestamp == timestamp

    def test_event_to_dict(self):
        """to_dict returns proper dictionary representation."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_PROGRESS,
            user_id="user-456",
            data={"progress": 50},
            timestamp="2024-01-15T10:30:00Z",
        )
        result = event.to_dict()
        assert result["type"] == "inbox_sync_progress"
        assert result["user_id"] == "user-456"
        assert result["data"]["progress"] == 50
        assert result["timestamp"] == "2024-01-15T10:30:00Z"

    def test_event_to_json(self):
        """to_json returns valid JSON string."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_COMPLETE,
            user_id="user-789",
            data={"messages_synced": 150},
            timestamp="2024-01-15T10:30:00Z",
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "inbox_sync_complete"
        assert parsed["user_id"] == "user-789"
        assert parsed["data"]["messages_synced"] == 150

    def test_event_with_string_type_in_to_dict(self):
        """to_dict handles string type values correctly."""
        event = InboxSyncEvent(
            type="custom_type",  # type: ignore
            user_id="user-123",
            data={},
        )
        result = event.to_dict()
        assert result["type"] == "custom_type"


# ===========================================================================
# Test InboxSyncEmitter - Subscription Management
# ===========================================================================


class TestInboxSyncEmitterSubscriptions:
    """Tests for InboxSyncEmitter subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_new_user(self, emitter, mock_ws):
        """Subscribe adds websocket to user's subscription set."""
        await emitter.subscribe("user-123", mock_ws)
        assert "user-123" in emitter._subscriptions
        assert mock_ws in emitter._subscriptions["user-123"]

    @pytest.mark.asyncio
    async def test_subscribe_multiple_connections_same_user(self, emitter, multiple_mock_ws):
        """Multiple connections can subscribe to same user."""
        for ws in multiple_mock_ws:
            await emitter.subscribe("user-123", ws)
        assert len(emitter._subscriptions["user-123"]) == 3

    @pytest.mark.asyncio
    async def test_subscribe_different_users(self, emitter, multiple_mock_ws):
        """Different users have separate subscription sets."""
        await emitter.subscribe("user-1", multiple_mock_ws[0])
        await emitter.subscribe("user-2", multiple_mock_ws[1])
        await emitter.subscribe("user-3", multiple_mock_ws[2])

        assert len(emitter._subscriptions) == 3
        assert multiple_mock_ws[0] in emitter._subscriptions["user-1"]
        assert multiple_mock_ws[1] in emitter._subscriptions["user-2"]

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_connection(self, emitter, mock_ws):
        """Unsubscribe removes the websocket from subscriptions."""
        await emitter.subscribe("user-123", mock_ws)
        await emitter.unsubscribe("user-123", mock_ws)
        assert "user-123" not in emitter._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_preserves_other_connections(self, emitter, multiple_mock_ws):
        """Unsubscribe only removes specific connection."""
        for ws in multiple_mock_ws:
            await emitter.subscribe("user-123", ws)

        await emitter.unsubscribe("user-123", multiple_mock_ws[0])

        assert len(emitter._subscriptions["user-123"]) == 2
        assert multiple_mock_ws[0] not in emitter._subscriptions["user-123"]
        assert multiple_mock_ws[1] in emitter._subscriptions["user-123"]

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_user(self, emitter, mock_ws):
        """Unsubscribe handles nonexistent user gracefully."""
        await emitter.unsubscribe("nonexistent", mock_ws)
        # Should not raise

    @pytest.mark.asyncio
    async def test_unsubscribe_last_connection_removes_user_entry(self, emitter, mock_ws):
        """Removing last connection cleans up user entry."""
        await emitter.subscribe("user-123", mock_ws)
        await emitter.unsubscribe("user-123", mock_ws)
        assert "user-123" not in emitter._subscriptions


# ===========================================================================
# Test InboxSyncEmitter - Event Emission
# ===========================================================================


class TestInboxSyncEmitterEmission:
    """Tests for InboxSyncEmitter event emission."""

    @pytest.mark.asyncio
    async def test_emit_sends_to_subscribed_client(self, emitter, mock_ws):
        """Emit sends event to subscribed client."""
        await emitter.subscribe("user-123", mock_ws)
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={"total_messages": 100},
        )

        sent_count = await emitter.emit(event)

        assert sent_count == 1
        mock_ws.send.assert_called_once()
        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "inbox_sync_start"

    @pytest.mark.asyncio
    async def test_emit_sends_to_all_user_connections(self, emitter, multiple_mock_ws):
        """Emit sends event to all user's connections."""
        for ws in multiple_mock_ws:
            await emitter.subscribe("user-123", ws)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_PROGRESS,
            user_id="user-123",
            data={"progress": 50},
        )

        sent_count = await emitter.emit(event)

        assert sent_count == 3
        for ws in multiple_mock_ws:
            ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_returns_zero_for_no_subscribers(self, emitter):
        """Emit returns 0 when no subscribers for user."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="nonexistent-user",
            data={},
        )
        sent_count = await emitter.emit(event)
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_emit_cleans_up_dead_connections(self, emitter, mock_ws, mock_ws_closed):
        """Emit removes dead connections after failed send."""
        await emitter.subscribe("user-123", mock_ws)
        await emitter.subscribe("user-123", mock_ws_closed)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
        )

        sent_count = await emitter.emit(event)

        assert sent_count == 1
        assert mock_ws_closed not in emitter._subscriptions["user-123"]
        assert mock_ws in emitter._subscriptions["user-123"]

    @pytest.mark.asyncio
    async def test_emit_sync_start(self, emitter, mock_ws):
        """emit_sync_start sends correct event."""
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_sync_start(
            user_id="user-123",
            total_messages=250,
            phase="Fetching message headers...",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "inbox_sync_start"
        assert sent_message["data"]["total_messages"] == 250
        assert sent_message["data"]["phase"] == "Fetching message headers..."

    @pytest.mark.asyncio
    async def test_emit_sync_progress(self, emitter, mock_ws):
        """emit_sync_progress sends correct event."""
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_sync_progress(
            user_id="user-123",
            progress=75,
            messages_synced=150,
            total_messages=200,
            phase="Downloading attachments...",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "inbox_sync_progress"
        assert sent_message["data"]["progress"] == 75
        assert sent_message["data"]["messages_synced"] == 150
        assert sent_message["data"]["total_messages"] == 200

    @pytest.mark.asyncio
    async def test_emit_sync_complete(self, emitter, mock_ws):
        """emit_sync_complete sends correct event."""
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_sync_complete(user_id="user-123", messages_synced=200)

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "inbox_sync_complete"
        assert sent_message["data"]["messages_synced"] == 200

    @pytest.mark.asyncio
    async def test_emit_sync_error(self, emitter, mock_ws):
        """emit_sync_error sends correct event."""
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_sync_error(
            user_id="user-123",
            error="Gmail API rate limit exceeded",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "inbox_sync_error"
        assert sent_message["data"]["error"] == "Gmail API rate limit exceeded"

    @pytest.mark.asyncio
    async def test_emit_new_priority_email(self, emitter, mock_ws):
        """emit_new_priority_email sends correct event."""
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_new_priority_email(
            user_id="user-123",
            email_id="email-abc-123",
            subject="Urgent: Board Meeting Tomorrow",
            from_address="ceo@company.com",
            priority="urgent",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert sent_message["type"] == "new_priority_email"
        assert sent_message["data"]["email"]["id"] == "email-abc-123"
        assert sent_message["data"]["email"]["subject"] == "Urgent: Board Meeting Tomorrow"
        assert sent_message["data"]["email"]["from_address"] == "ceo@company.com"
        assert sent_message["data"]["email"]["priority"] == "urgent"


# ===========================================================================
# Test InboxSyncEmitter - Callbacks
# ===========================================================================


class TestInboxSyncEmitterCallbacks:
    """Tests for InboxSyncEmitter callback functionality."""

    @pytest.mark.asyncio
    async def test_add_callback_receives_events(self, emitter):
        """Added callback receives emitted events."""
        received_events = []

        def callback(event):
            received_events.append(event)

        emitter.add_callback(callback)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
        )
        await emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0].type == InboxSyncEventType.SYNC_START

    @pytest.mark.asyncio
    async def test_multiple_callbacks_all_receive_events(self, emitter):
        """Multiple callbacks all receive events."""
        received_1 = []
        received_2 = []

        emitter.add_callback(lambda e: received_1.append(e))
        emitter.add_callback(lambda e: received_2.append(e))

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_PROGRESS,
            user_id="user-123",
            data={"progress": 50},
        )
        await emitter.emit(event)

        assert len(received_1) == 1
        assert len(received_2) == 1

    @pytest.mark.asyncio
    async def test_remove_callback(self, emitter):
        """Removed callback no longer receives events."""
        received_events = []

        def callback(event):
            received_events.append(event)

        emitter.add_callback(callback)
        emitter.remove_callback(callback)

        await emitter.emit(
            InboxSyncEvent(
                type=InboxSyncEventType.SYNC_START,
                user_id="user-123",
                data={},
            )
        )

        assert len(received_events) == 0

    @pytest.mark.asyncio
    async def test_callback_error_does_not_break_emission(self, emitter, mock_ws):
        """Callback error doesn't prevent event emission."""
        await emitter.subscribe("user-123", mock_ws)

        def failing_callback(event):
            raise RuntimeError("Callback failed")

        emitter.add_callback(failing_callback)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
        )

        # Should not raise
        sent_count = await emitter.emit(event)
        assert sent_count == 1
        mock_ws.send.assert_called_once()


# ===========================================================================
# Test InboxSyncEmitter - Concurrent Updates
# ===========================================================================


class TestInboxSyncEmitterConcurrency:
    """Tests for InboxSyncEmitter concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_subscriptions(self, emitter):
        """Concurrent subscriptions are handled safely."""
        websockets = [MagicMock(send=AsyncMock()) for _ in range(10)]

        async def subscribe_ws(ws, i):
            await emitter.subscribe(f"user-{i % 3}", ws)

        await asyncio.gather(*[subscribe_ws(ws, i) for i, ws in enumerate(websockets)])

        # Verify all subscriptions were added
        total_subscriptions = sum(len(subs) for subs in emitter._subscriptions.values())
        assert total_subscriptions == 10

    @pytest.mark.asyncio
    async def test_concurrent_emissions(self, emitter, multiple_mock_ws):
        """Concurrent emissions don't interfere with each other."""
        for ws in multiple_mock_ws:
            await emitter.subscribe("user-123", ws)

        events = [
            InboxSyncEvent(
                type=InboxSyncEventType.SYNC_PROGRESS,
                user_id="user-123",
                data={"progress": i * 10},
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*[emitter.emit(event) for event in events])

        assert all(count == 3 for count in results)

    @pytest.mark.asyncio
    async def test_concurrent_subscribe_and_emit(self, emitter):
        """Concurrent subscribe and emit operations are safe."""
        websocket = MagicMock(send=AsyncMock())

        async def subscribe_and_emit():
            await emitter.subscribe("user-123", websocket)
            event = InboxSyncEvent(
                type=InboxSyncEventType.SYNC_START,
                user_id="user-123",
                data={},
            )
            await emitter.emit(event)

        async def unsubscribe():
            await asyncio.sleep(0.01)
            await emitter.unsubscribe("user-123", websocket)

        await asyncio.gather(subscribe_and_emit(), unsubscribe())
        # Should not raise

    @pytest.mark.asyncio
    async def test_concurrent_unsubscribe_during_emit(self, emitter):
        """Unsubscribe during emit doesn't cause errors."""
        websockets = [MagicMock(send=AsyncMock()) for _ in range(5)]
        for ws in websockets:
            await emitter.subscribe("user-123", ws)

        async def emit_event():
            for _ in range(5):
                event = InboxSyncEvent(
                    type=InboxSyncEventType.SYNC_PROGRESS,
                    user_id="user-123",
                    data={"progress": 50},
                )
                await emitter.emit(event)

        async def unsubscribe_all():
            await asyncio.sleep(0.001)
            for ws in websockets:
                await emitter.unsubscribe("user-123", ws)

        await asyncio.gather(emit_event(), unsubscribe_all())
        # Should complete without error


# ===========================================================================
# Test InboxSyncEmitter - Error Handling
# ===========================================================================


class TestInboxSyncEmitterErrorHandling:
    """Tests for InboxSyncEmitter error handling."""

    @pytest.mark.asyncio
    async def test_handles_oserror_on_send(self, emitter):
        """OSError on send is handled gracefully."""
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock(side_effect=OSError("Network error"))
        await emitter.subscribe("user-123", mock_ws)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
        )

        sent_count = await emitter.emit(event)
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_handles_runtime_error_on_send(self, emitter):
        """RuntimeError on send is handled gracefully."""
        mock_ws = MagicMock()
        mock_ws.send = AsyncMock(side_effect=RuntimeError("WebSocket closed"))
        await emitter.subscribe("user-123", mock_ws)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
        )

        sent_count = await emitter.emit(event)
        assert sent_count == 0
        # Dead connection should be cleaned up
        assert mock_ws not in emitter._subscriptions.get("user-123", set())

    @pytest.mark.asyncio
    async def test_partial_failure_still_sends_to_healthy_clients(self, emitter):
        """Partial failure doesn't prevent sending to healthy clients."""
        healthy_ws = MagicMock(send=AsyncMock())
        failing_ws = MagicMock(send=AsyncMock(side_effect=ConnectionError("Closed")))

        await emitter.subscribe("user-123", healthy_ws)
        await emitter.subscribe("user-123", failing_ws)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_COMPLETE,
            user_id="user-123",
            data={"messages_synced": 100},
        )

        sent_count = await emitter.emit(event)
        assert sent_count == 1
        healthy_ws.send.assert_called_once()


# ===========================================================================
# Test Global Emitter and Convenience Functions
# ===========================================================================


class TestGlobalEmitter:
    """Tests for global emitter instance and convenience functions."""

    def test_get_inbox_sync_emitter_creates_instance(self):
        """get_inbox_sync_emitter creates a singleton instance."""
        emitter = get_inbox_sync_emitter()
        assert emitter is not None
        assert isinstance(emitter, InboxSyncEmitter)

    def test_get_inbox_sync_emitter_returns_same_instance(self):
        """get_inbox_sync_emitter returns the same instance on subsequent calls."""
        emitter1 = get_inbox_sync_emitter()
        emitter2 = get_inbox_sync_emitter()
        assert emitter1 is emitter2

    @pytest.mark.asyncio
    async def test_emit_sync_start_convenience(self):
        """emit_sync_start convenience function works."""
        emitter = get_inbox_sync_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        sent = await emit_sync_start("user-123", total_messages=100)
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_sync_progress_convenience(self):
        """emit_sync_progress convenience function works."""
        emitter = get_inbox_sync_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        sent = await emit_sync_progress(
            "user-123",
            progress=50,
            messages_synced=50,
            total_messages=100,
        )
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_sync_complete_convenience(self):
        """emit_sync_complete convenience function works."""
        emitter = get_inbox_sync_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        sent = await emit_sync_complete("user-123", messages_synced=100)
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_sync_error_convenience(self):
        """emit_sync_error convenience function works."""
        emitter = get_inbox_sync_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        sent = await emit_sync_error("user-123", error="API error")
        assert sent == 1

    @pytest.mark.asyncio
    async def test_emit_new_priority_email_convenience(self):
        """emit_new_priority_email convenience function works."""
        emitter = get_inbox_sync_emitter()
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        sent = await emit_new_priority_email(
            "user-123",
            email_id="email-1",
            subject="Test",
            from_address="test@example.com",
            priority="high",
        )
        assert sent == 1


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestInboxSyncEdgeCases:
    """Tests for edge cases in inbox sync functionality."""

    @pytest.mark.asyncio
    async def test_emit_to_empty_subscription_set(self, emitter):
        """Emit to user with empty subscription set (after all unsubscribe)."""
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)
        await emitter.unsubscribe("user-123", mock_ws)

        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id="user-123",
            data={},
        )
        sent_count = await emitter.emit(event)
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_duplicate_subscription_same_websocket(self, emitter, mock_ws):
        """Duplicate subscription of same websocket is idempotent."""
        await emitter.subscribe("user-123", mock_ws)
        await emitter.subscribe("user-123", mock_ws)

        # Should only have one subscription
        assert len(emitter._subscriptions["user-123"]) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_already_unsubscribed(self, emitter, mock_ws):
        """Unsubscribing an already unsubscribed websocket is safe."""
        await emitter.subscribe("user-123", mock_ws)
        await emitter.unsubscribe("user-123", mock_ws)
        await emitter.unsubscribe("user-123", mock_ws)
        # Should not raise

    def test_event_with_complex_data(self):
        """Event handles complex nested data structures."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.NEW_PRIORITY_EMAIL,
            user_id="user-123",
            data={
                "email": {
                    "id": "email-1",
                    "subject": "Test",
                    "attachments": [{"name": "file.pdf", "size": 1024}],
                    "labels": ["important", "work"],
                },
            },
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"]["email"]["attachments"][0]["name"] == "file.pdf"

    @pytest.mark.asyncio
    async def test_emit_with_unicode_content(self, emitter, mock_ws):
        """Emit handles unicode content correctly."""
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_new_priority_email(
            user_id="user-123",
            email_id="email-1",
            subject="Meeting with CEO: \u65e5\u672c\u8a9e\u30c6\u30b9\u30c8",
            from_address="test@example.com",
            priority="high",
        )

        sent_message = json.loads(mock_ws.send.call_args[0][0])
        assert "\u65e5\u672c\u8a9e" in sent_message["data"]["email"]["subject"]

    @pytest.mark.asyncio
    async def test_remove_nonexistent_callback(self, emitter):
        """Removing a non-existent callback is safe."""

        def callback(e):
            pass

        emitter.remove_callback(callback)
        # Should not raise


# ===========================================================================
# Test Message Sync Callbacks Integration
# ===========================================================================


class TestMessageSyncCallbacksIntegration:
    """Tests for message sync callback integration patterns."""

    @pytest.mark.asyncio
    async def test_sync_lifecycle_with_callbacks(self, emitter):
        """Full sync lifecycle with callback tracking."""
        lifecycle_events = []

        def track_lifecycle(event):
            lifecycle_events.append(event.type)

        emitter.add_callback(track_lifecycle)
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        # Simulate full sync lifecycle
        await emitter.emit_sync_start("user-123", total_messages=100)
        for i in range(1, 5):
            await emitter.emit_sync_progress(
                "user-123",
                progress=i * 25,
                messages_synced=i * 25,
                total_messages=100,
            )
        await emitter.emit_sync_complete("user-123", messages_synced=100)

        assert lifecycle_events == [
            InboxSyncEventType.SYNC_START,
            InboxSyncEventType.SYNC_PROGRESS,
            InboxSyncEventType.SYNC_PROGRESS,
            InboxSyncEventType.SYNC_PROGRESS,
            InboxSyncEventType.SYNC_PROGRESS,
            InboxSyncEventType.SYNC_COMPLETE,
        ]

    @pytest.mark.asyncio
    async def test_sync_failure_lifecycle(self, emitter):
        """Sync failure lifecycle with callback tracking."""
        lifecycle_events = []

        def track_lifecycle(event):
            lifecycle_events.append((event.type, event.data.get("error")))

        emitter.add_callback(track_lifecycle)
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_sync_start("user-123", total_messages=100)
        await emitter.emit_sync_progress(
            "user-123", progress=30, messages_synced=30, total_messages=100
        )
        await emitter.emit_sync_error("user-123", error="OAuth token expired")

        assert lifecycle_events[-1] == (InboxSyncEventType.SYNC_ERROR, "OAuth token expired")

    @pytest.mark.asyncio
    async def test_priority_email_during_sync(self, emitter):
        """Priority email notification during sync."""
        events_received = []

        def track_all(event):
            events_received.append(event.type)

        emitter.add_callback(track_all)
        mock_ws = MagicMock(send=AsyncMock())
        await emitter.subscribe("user-123", mock_ws)

        await emitter.emit_sync_start("user-123", total_messages=100)
        await emitter.emit_sync_progress(
            "user-123", progress=50, messages_synced=50, total_messages=100
        )
        # Priority email found during sync
        await emitter.emit_new_priority_email(
            "user-123",
            email_id="urgent-1",
            subject="Urgent: Server Down",
            from_address="alerts@company.com",
            priority="urgent",
        )
        await emitter.emit_sync_complete("user-123", messages_synced=100)

        assert InboxSyncEventType.NEW_PRIORITY_EMAIL in events_received
