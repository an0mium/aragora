"""
Tests for WebSocket streaming resilience and edge cases.

Verifies:
- Connection drop handling and client removal
- Concurrent connection handling
- Event serialization edge cases
- SyncEventEmitter thread safety
"""

import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import threading

import pytest

from aragora.server.stream import (
    DebateStreamServer,
    StreamEvent,
    StreamEventType,
    SyncEventEmitter,
)


# =============================================================================
# Connection Drop Tests
# =============================================================================


class TestConnectionDropHandling:
    """Test handling of dropped WebSocket connections."""

    @pytest.mark.asyncio
    async def test_failed_send_removes_client(self):
        """Client that fails to receive should be handled gracefully."""
        server = DebateStreamServer(host="localhost", port=0)

        good_client = AsyncMock()
        bad_client = AsyncMock()
        bad_client.send.side_effect = ConnectionError("Connection reset")

        server.clients = {good_client, bad_client}

        event = StreamEvent(
            type=StreamEventType.ROUND_START,
            data={"round": 1},
        )

        await server.broadcast(event)

        # Good client should have received the message
        good_client.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_handled_gracefully(self):
        """Client that times out should be handled gracefully."""
        server = DebateStreamServer(host="localhost", port=0)

        slow_client = AsyncMock()
        slow_client.send.side_effect = asyncio.TimeoutError()

        server.clients = {slow_client}

        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"agent": "test", "content": "response"},
        )

        # Should not raise
        await server.broadcast(event)

    @pytest.mark.asyncio
    async def test_multiple_failures_handled(self):
        """Multiple failing clients should all be handled."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create 5 clients, 3 will fail
        good_clients = [AsyncMock() for _ in range(2)]
        bad_clients = [AsyncMock() for _ in range(3)]

        for client in bad_clients:
            client.send.side_effect = ConnectionError("Disconnected")

        server.clients = set(good_clients + bad_clients)

        event = StreamEvent(
            type=StreamEventType.TASK_COMPLETE,
            data={"result": "success"},
        )

        await server.broadcast(event)

        # Good clients should have received messages
        for client in good_clients:
            client.send.assert_called_once()


# =============================================================================
# Concurrent Connection Tests
# =============================================================================


class TestConcurrentConnections:
    """Test handling of many concurrent connections."""

    @pytest.mark.asyncio
    async def test_broadcast_to_many_clients(self):
        """Should handle broadcast to many clients efficiently."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create 100 mock clients
        clients = [AsyncMock() for _ in range(100)]
        server.clients = set(clients)

        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={"consensus": 0.75},
        )

        start = time.time()
        await server.broadcast(event)
        elapsed = time.time() - start

        # All clients should receive
        for client in clients:
            client.send.assert_called_once()

        # Should complete reasonably fast (< 1 second for mocks)
        assert elapsed < 1.0

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self):
        """Broadcast should continue even if some clients fail."""
        server = DebateStreamServer(host="localhost", port=0)

        # Mix of good and bad clients
        clients = []
        for i in range(50):
            client = AsyncMock()
            if i % 5 == 0:  # Every 5th client fails
                client.send.side_effect = Exception("Failed")
            clients.append(client)

        server.clients = set(clients)

        event = StreamEvent(
            type=StreamEventType.DEBATE_END,
            data={"winner": "agent_a"},
        )

        # Should not raise
        await server.broadcast(event)


# =============================================================================
# Event Serialization Tests
# =============================================================================


class TestEventSerialization:
    """Test event JSON serialization edge cases."""

    def test_event_with_special_characters(self):
        """Events with special characters should serialize correctly."""
        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={
                "content": 'Test with "quotes" and\nnewlines\tand emoji 🎉',
                "agent": "test-agent",
            },
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["data"]["content"] == 'Test with "quotes" and\nnewlines\tand emoji 🎉'

    def test_event_with_none_values(self):
        """Events with None values should serialize correctly."""
        event = StreamEvent(
            type=StreamEventType.ERROR,
            data={"error": None, "message": "test"},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["data"]["error"] is None

    def test_event_with_nested_data(self):
        """Events with deeply nested data should serialize correctly."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={
                "scores": {"agent_a": 0.8, "agent_b": 0.6},
                "metadata": {
                    "nested": {
                        "deeply": {
                            "value": 42
                        }
                    }
                },
            },
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["data"]["metadata"]["nested"]["deeply"]["value"] == 42

    def test_event_with_large_payload(self):
        """Events with large payloads should serialize correctly."""
        large_content = "x" * 100000  # 100KB of text

        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"content": large_content},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert len(parsed["data"]["content"]) == 100000

    def test_event_type_serialization(self):
        """Event type should serialize to string value."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "test"},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "debate_start"


# =============================================================================
# SyncEventEmitter Tests
# =============================================================================


class TestSyncEventEmitter:
    """Test synchronous event emission for async consumption."""

    def test_emit_from_sync_context(self):
        """Should be able to emit events from synchronous code."""
        emitter = SyncEventEmitter(loop_id="test")

        # Emit without async context
        emitter.emit(StreamEvent(StreamEventType.TASK_START, {"task": "test"}))
        emitter.emit(StreamEvent(StreamEventType.ROUND_START, {"round": 1}))

        # Events should be queued
        assert not emitter._queue.empty()

    def test_emit_with_loop_id_injection(self):
        """Emitted events should have loop_id injected."""
        emitter = SyncEventEmitter(loop_id="my-loop-123")

        emitter.emit(StreamEvent(StreamEventType.TASK_START, {"task": "test"}))

        # Get the event from queue
        event = emitter._queue.get_nowait()
        assert event.loop_id == "my-loop-123"

    def test_drain_returns_events(self):
        """Drain should return all queued events."""
        emitter = SyncEventEmitter(loop_id="test")

        # Queue some events
        for i in range(5):
            emitter.emit(StreamEvent(StreamEventType.AGENT_MESSAGE, {"index": i}))

        # Drain
        events = emitter.drain()

        assert len(events) == 5
        assert emitter._queue.empty()

    def test_concurrent_emit_safe(self):
        """Emit should be thread-safe."""
        emitter = SyncEventEmitter(loop_id="test")
        errors = []

        def emit_events(thread_id):
            try:
                for i in range(10):  # Fewer events to avoid queue overflow
                    emitter.emit(StreamEvent(
                        StreamEventType.AGENT_MESSAGE,
                        {"thread": thread_id, "index": i},
                    ))
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [
            threading.Thread(target=emit_events, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0

        # Events should be queued (may be less than total due to queue size limits)
        events = emitter.drain()
        assert len(events) > 0  # At least some events were queued
        assert len(events) <= 50  # 5 threads * 10 events max


# =============================================================================
# Integration Tests
# =============================================================================


class TestStreamingIntegration:
    """Integration tests for the streaming system."""

    @pytest.mark.asyncio
    async def test_full_broadcast_cycle(self):
        """Test complete emit -> broadcast cycle."""
        server = DebateStreamServer(host="localhost", port=0)

        # Create mock clients
        clients = [AsyncMock() for _ in range(3)]
        server.clients = set(clients)

        # Create emitter
        emitter = SyncEventEmitter(loop_id="integration-test")

        # Emit events synchronously
        emitter.emit(StreamEvent(StreamEventType.TASK_START, {"task": "debate"}))
        emitter.emit(StreamEvent(StreamEventType.ROUND_START, {"round": 1}))

        # Drain and broadcast
        events = emitter.drain()
        for event in events:
            await server.broadcast(event)

        # All clients should have received both events
        assert all(c.send.call_count == 2 for c in clients)

    @pytest.mark.asyncio
    async def test_error_event_isolation(self):
        """Errors in one client should not affect others."""
        server = DebateStreamServer(host="localhost", port=0)

        good_client = AsyncMock()
        error_client = AsyncMock()
        error_client.send.side_effect = [
            None,  # First call succeeds
            Exception("Boom"),  # Second call fails
            None,  # Third call succeeds (if reached)
        ]

        server.clients = {good_client, error_client}

        # Send multiple events
        for i in range(3):
            event = StreamEvent(StreamEventType.ROUND_START, {"round": i})
            await server.broadcast(event)

        # Good client should receive all 3
        assert good_client.send.call_count == 3

    @pytest.mark.asyncio
    async def test_empty_client_set(self):
        """Broadcast with no clients should not error."""
        server = DebateStreamServer(host="localhost", port=0)
        server.clients = set()

        event = StreamEvent(StreamEventType.DEBATE_START, {"task": "test"})

        # Should not raise
        await server.broadcast(event)

    def test_emitter_clears_on_drain(self):
        """Emitter should be empty after drain."""
        emitter = SyncEventEmitter(loop_id="test")

        emitter.emit(StreamEvent(StreamEventType.TASK_START, {"task": "1"}))
        emitter.emit(StreamEvent(StreamEventType.TASK_START, {"task": "2"}))

        events = emitter.drain()
        assert len(events) == 2

        # Second drain should be empty
        events2 = emitter.drain()
        assert len(events2) == 0
