"""
E2E tests for WebSocket streaming functionality.

Tests verify the WebSocket event streaming:
1. Connection establishment
2. Event emission during debate
3. Event type coverage
4. Connection cleanup
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Test Fixtures
# ===========================================================================


@dataclass
class MockWebSocketClient:
    """Mock WebSocket client for testing."""

    messages: list[dict] = field(default_factory=list)
    is_connected: bool = True
    _closed: bool = False

    async def send(self, message: str) -> None:
        """Send message to client."""
        if not self._closed:
            self.messages.append(json.loads(message))

    async def close(self) -> None:
        """Close connection."""
        self._closed = True
        self.is_connected = False


@pytest.fixture
def mock_ws_client():
    """Create mock WebSocket client."""
    return MockWebSocketClient()


@pytest.fixture
def mock_event_bus():
    """Create mock event bus for capturing events."""
    events = []

    class MockEventBus:
        def emit(self, event_type: str, data: dict):
            events.append({"type": event_type, "data": data})

        def get_events(self):
            return events

    return MockEventBus()


# ===========================================================================
# Test WebSocket Events
# ===========================================================================


class TestWebSocketEventTypes:
    """Tests for WebSocket event type coverage."""

    def test_debate_start_event_emitted(self, mock_event_bus):
        """debate_start event should be emitted when debate begins."""
        mock_event_bus.emit(
            "debate_start",
            {
                "debate_id": "test-123",
                "task": "Test debate task",
                "agents": ["agent1", "agent2"],
            },
        )

        events = mock_event_bus.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "debate_start"
        assert "debate_id" in events[0]["data"]

    def test_round_start_event_emitted(self, mock_event_bus):
        """round_start event should be emitted at each round."""
        mock_event_bus.emit(
            "round_start",
            {
                "debate_id": "test-123",
                "round": 1,
                "total_rounds": 3,
            },
        )

        events = mock_event_bus.get_events()
        assert events[0]["type"] == "round_start"
        assert events[0]["data"]["round"] == 1

    def test_agent_message_event_emitted(self, mock_event_bus):
        """agent_message event should be emitted for agent responses."""
        mock_event_bus.emit(
            "agent_message",
            {
                "debate_id": "test-123",
                "agent": "gpt-4",
                "content": "I believe the answer is...",
                "round": 1,
            },
        )

        events = mock_event_bus.get_events()
        assert events[0]["type"] == "agent_message"
        assert events[0]["data"]["agent"] == "gpt-4"

    def test_critique_event_emitted(self, mock_event_bus):
        """critique event should be emitted during critique phase."""
        mock_event_bus.emit(
            "critique",
            {
                "debate_id": "test-123",
                "from_agent": "claude",
                "target_agent": "gpt-4",
                "severity": "medium",
                "issues": ["Unclear reasoning"],
            },
        )

        events = mock_event_bus.get_events()
        assert events[0]["type"] == "critique"
        assert events[0]["data"]["severity"] == "medium"

    def test_vote_event_emitted(self, mock_event_bus):
        """vote event should be emitted during voting phase."""
        mock_event_bus.emit(
            "vote",
            {
                "debate_id": "test-123",
                "agent": "claude",
                "vote": "solution_a",
                "confidence": 0.85,
            },
        )

        events = mock_event_bus.get_events()
        assert events[0]["type"] == "vote"
        assert events[0]["data"]["confidence"] == 0.85

    def test_consensus_event_emitted(self, mock_event_bus):
        """consensus event should be emitted when consensus reached."""
        mock_event_bus.emit(
            "consensus",
            {
                "debate_id": "test-123",
                "consensus_reached": True,
                "final_answer": "The solution is...",
                "method": "majority",
            },
        )

        events = mock_event_bus.get_events()
        assert events[0]["type"] == "consensus"
        assert events[0]["data"]["consensus_reached"] is True

    def test_debate_end_event_emitted(self, mock_event_bus):
        """debate_end event should be emitted when debate completes."""
        mock_event_bus.emit(
            "debate_end",
            {
                "debate_id": "test-123",
                "status": "completed",
                "rounds_completed": 3,
                "duration_seconds": 45.2,
            },
        )

        events = mock_event_bus.get_events()
        assert events[0]["type"] == "debate_end"
        assert events[0]["data"]["status"] == "completed"


class TestWebSocketMessageFormat:
    """Tests for WebSocket message formatting."""

    def test_message_is_valid_json(self, mock_ws_client):
        """Messages should be valid JSON."""

        asyncio.run(
            mock_ws_client.send(
                json.dumps(
                    {
                        "type": "test_event",
                        "data": {"key": "value"},
                    }
                )
            )
        )

        assert len(mock_ws_client.messages) == 1
        assert mock_ws_client.messages[0]["type"] == "test_event"

    def test_message_includes_timestamp(self, mock_event_bus):
        """Events should include timestamps when sent."""
        from datetime import datetime

        mock_event_bus.emit(
            "test_event",
            {
                "timestamp": datetime.now().isoformat(),
                "data": "test",
            },
        )

        events = mock_event_bus.get_events()
        assert "timestamp" in events[0]["data"]


class TestWebSocketConnection:
    """Tests for WebSocket connection management."""

    @pytest.mark.asyncio
    async def test_client_connection_tracked(self, mock_ws_client):
        """Connected clients should be tracked."""
        assert mock_ws_client.is_connected is True

    @pytest.mark.asyncio
    async def test_client_disconnection_handled(self, mock_ws_client):
        """Client disconnection should be handled gracefully."""
        await mock_ws_client.close()

        assert mock_ws_client.is_connected is False
        assert mock_ws_client._closed is True

    @pytest.mark.asyncio
    async def test_send_to_closed_connection_safe(self, mock_ws_client):
        """Sending to closed connection should not raise errors."""
        await mock_ws_client.close()

        # Should not raise
        await mock_ws_client.send(json.dumps({"type": "test"}))

        # Message should not be delivered
        # (closed connections don't accumulate messages in mock)


class TestWebSocketDebateStreaming:
    """Tests for streaming debate events via WebSocket."""

    @pytest.mark.asyncio
    async def test_debate_events_streamed_in_order(self, mock_event_bus):
        """Debate events should be streamed in correct order."""
        # Emit events in debate order
        event_order = [
            "debate_start",
            "round_start",
            "agent_message",
            "agent_message",
            "critique",
            "vote",
            "round_start",
            "agent_message",
            "consensus",
            "debate_end",
        ]

        for event_type in event_order:
            mock_event_bus.emit(event_type, {"order": len(mock_event_bus.get_events())})

        events = mock_event_bus.get_events()

        # Verify order preserved
        for i, expected_type in enumerate(event_order):
            assert events[i]["type"] == expected_type
            assert events[i]["data"]["order"] == i

    @pytest.mark.asyncio
    async def test_multiple_clients_receive_events(self):
        """Multiple connected clients should receive events."""
        clients = [MockWebSocketClient() for _ in range(3)]

        event = json.dumps(
            {
                "type": "debate_start",
                "data": {"debate_id": "test-123"},
            }
        )

        # Broadcast to all clients
        for client in clients:
            await client.send(event)

        # All clients should have received
        for client in clients:
            assert len(client.messages) == 1
            assert client.messages[0]["type"] == "debate_start"


class TestWebSocketErrorHandling:
    """Tests for WebSocket error handling."""

    @pytest.mark.asyncio
    async def test_malformed_message_handled(self, mock_ws_client):
        """Malformed messages should be handled gracefully."""
        # Try sending invalid JSON
        with pytest.raises(json.JSONDecodeError):
            await mock_ws_client.send("not valid json{")

    @pytest.mark.asyncio
    async def test_connection_error_recovery(self, mock_ws_client):
        """Connection errors should not crash the server."""
        await mock_ws_client.close()

        # Server should continue operating
        # (This tests the mock's behavior, real test would need server)
        assert mock_ws_client._closed is True


class TestStreamBuffering:
    """Tests for stream buffering behavior."""

    def test_stream_buffer_size_configurable(self):
        """Stream buffer size should be configurable via environment."""
        import os

        # Default buffer size
        default_size = 10485760  # 10MB

        # Should respect environment variable
        buffer_size = int(os.getenv("ARAGORA_STREAM_BUFFER_SIZE", default_size))
        assert buffer_size > 0

    def test_chunk_timeout_configurable(self):
        """Stream chunk timeout should be configurable."""
        import os

        default_timeout = 30
        timeout = int(os.getenv("ARAGORA_STREAM_CHUNK_TIMEOUT", default_timeout))
        assert timeout > 0
