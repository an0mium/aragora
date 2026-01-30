"""
Shared fixtures for WebSocket streaming tests.

Provides mock objects for testing:
- MockWebSocketConnection - Simulates aiohttp/websockets WebSocket
- MockStreamEmitter - Captures emitted events for verification
- MockAuthConfig - Controls authentication behavior
- Event creation helpers
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.events.types import StreamEvent, StreamEventType


# ===========================================================================
# Mock WebSocket Connection
# ===========================================================================


class MockWebSocketConnection:
    """Mock WebSocket connection for testing.

    Simulates a WebSocket connection with send/receive capabilities.
    Tracks all sent messages for verification.
    """

    def __init__(
        self,
        remote_address: tuple[str, int] = ("127.0.0.1", 12345),
        headers: dict[str, str] | None = None,
    ):
        self.messages: list[str | dict] = []
        self.closed: bool = False
        self.close_code: int | None = None
        self.close_reason: str = ""
        self.remote_address = remote_address
        self._receive_queue: asyncio.Queue[dict] = asyncio.Queue()
        self._headers = headers or {}

        # Support both websockets and aiohttp APIs
        self.request = MagicMock()
        self.request.headers = self._headers
        self.request_headers = self._headers

    async def send(self, message: str) -> None:
        """Send a message (simulates WebSocket.send)."""
        if self.closed:
            raise ConnectionError("WebSocket is closed")
        self.messages.append(message)

    async def send_json(self, data: dict) -> None:
        """Send JSON data (aiohttp style)."""
        await self.send(json.dumps(data))

    async def send_str(self, message: str) -> None:
        """Send string (aiohttp style)."""
        await self.send(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close the WebSocket connection."""
        self.closed = True
        self.close_code = code
        self.close_reason = reason

    async def receive(self) -> dict:
        """Receive a message from the queue (for testing incoming messages)."""
        return await self._receive_queue.get()

    async def receive_json(self) -> dict:
        """Receive JSON message (aiohttp style)."""
        msg = await self.receive()
        if isinstance(msg, str):
            return json.loads(msg)
        return msg

    def queue_message(self, message: dict | str) -> None:
        """Queue a message to be received by the handler."""
        self._receive_queue.put_nowait(message)

    def get_sent_messages(self) -> list[dict]:
        """Get all sent messages parsed as JSON."""
        result = []
        for msg in self.messages:
            if isinstance(msg, str):
                try:
                    result.append(json.loads(msg))
                except json.JSONDecodeError:
                    result.append({"raw": msg})
            else:
                result.append(msg)
        return result

    def get_last_message(self) -> dict | None:
        """Get the last sent message parsed as JSON."""
        messages = self.get_sent_messages()
        return messages[-1] if messages else None

    def clear_messages(self) -> None:
        """Clear all recorded messages."""
        self.messages.clear()


# ===========================================================================
# Mock Stream Emitter
# ===========================================================================


class MockStreamEmitter:
    """Mock event emitter that captures all emitted events.

    Use this to verify that handlers emit the correct events
    during debate streaming operations.
    """

    def __init__(self):
        self.events: list[StreamEvent] = []
        self._event_dicts: list[dict] = []

    async def emit(self, event: StreamEvent) -> None:
        """Emit an event (async version)."""
        self.events.append(event)
        self._event_dicts.append(event.to_dict())

    def emit_sync(self, event: StreamEvent) -> None:
        """Emit an event (sync version for SyncEventEmitter)."""
        self.events.append(event)
        self._event_dicts.append(event.to_dict())

    def get_events_by_type(self, event_type: StreamEventType) -> list[StreamEvent]:
        """Get all events of a specific type."""
        return [e for e in self.events if e.type == event_type]

    def has_event(self, event_type: StreamEventType) -> bool:
        """Check if an event of the given type was emitted."""
        return any(e.type == event_type for e in self.events)

    def get_last_event(self) -> StreamEvent | None:
        """Get the last emitted event."""
        return self.events[-1] if self.events else None

    def get_last_event_of_type(self, event_type: StreamEventType) -> StreamEvent | None:
        """Get the last event of a specific type."""
        events = self.get_events_by_type(event_type)
        return events[-1] if events else None

    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()
        self._event_dicts.clear()

    @property
    def event_count(self) -> int:
        """Total number of emitted events."""
        return len(self.events)


# ===========================================================================
# Event Factory Helpers
# ===========================================================================


def create_debate_start_event(
    loop_id: str = "test-debate-1",
    task: str = "Test question",
    agents: list[str] | None = None,
) -> StreamEvent:
    """Create a DEBATE_START event for testing."""
    return StreamEvent(
        type=StreamEventType.DEBATE_START,
        data={
            "task": task,
            "agents": agents or ["claude", "gpt4", "gemini"],
        },
        loop_id=loop_id,
    )


def create_agent_message_event(
    loop_id: str = "test-debate-1",
    agent: str = "claude",
    content: str = "Test response",
    round: int = 1,
    role: str = "proposer",
) -> StreamEvent:
    """Create an AGENT_MESSAGE event for testing."""
    return StreamEvent(
        type=StreamEventType.AGENT_MESSAGE,
        data={
            "content": content,
            "role": role,
        },
        loop_id=loop_id,
        agent=agent,
        round=round,
    )


def create_consensus_event(
    loop_id: str = "test-debate-1",
    reached: bool = True,
    confidence: float = 0.85,
    answer: str = "Test consensus answer",
) -> StreamEvent:
    """Create a CONSENSUS event for testing."""
    return StreamEvent(
        type=StreamEventType.CONSENSUS,
        data={
            "reached": reached,
            "confidence": confidence,
            "answer": answer,
        },
        loop_id=loop_id,
    )


def create_debate_end_event(
    loop_id: str = "test-debate-1",
    duration: float = 45.0,
    rounds: int = 3,
) -> StreamEvent:
    """Create a DEBATE_END event for testing."""
    return StreamEvent(
        type=StreamEventType.DEBATE_END,
        data={
            "duration": duration,
            "rounds": rounds,
        },
        loop_id=loop_id,
    )


def create_round_start_event(
    loop_id: str = "test-debate-1",
    round: int = 1,
) -> StreamEvent:
    """Create a ROUND_START event for testing."""
    return StreamEvent(
        type=StreamEventType.ROUND_START,
        data={"round": round},
        loop_id=loop_id,
        round=round,
    )


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_ws():
    """Create a mock WebSocket connection."""
    return MockWebSocketConnection()


@pytest.fixture
def mock_ws_with_auth():
    """Create a mock WebSocket with Bearer token auth."""
    return MockWebSocketConnection(headers={"Authorization": "Bearer test-token-123"})


@pytest.fixture
def mock_emitter():
    """Create a mock stream emitter."""
    return MockStreamEmitter()


@pytest.fixture
def mock_auth_disabled():
    """Mock auth_config with authentication disabled."""
    with patch("aragora.server.auth.auth_config") as mock:
        mock.enabled = False
        mock.validate_token = MagicMock(return_value=True)
        yield mock


@pytest.fixture
def mock_auth_enabled():
    """Mock auth_config with authentication enabled."""
    with patch("aragora.server.auth.auth_config") as mock:
        mock.enabled = True
        mock.validate_token = MagicMock(return_value=True)
        yield mock


@pytest.fixture
def mock_auth_reject():
    """Mock auth_config that rejects all tokens."""
    with patch("aragora.server.auth.auth_config") as mock:
        mock.enabled = True
        mock.validate_token = MagicMock(return_value=False)
        yield mock


@pytest.fixture
def disable_rate_limiting(monkeypatch):
    """Disable WebSocket rate limiting for tests."""
    monkeypatch.setenv("ARAGORA_DISABLE_ALL_RATE_LIMITS", "true")


@pytest.fixture
def sample_debate_events():
    """Create a sequence of debate events for testing broadcast."""
    loop_id = "test-debate-1"
    return [
        create_debate_start_event(loop_id),
        create_round_start_event(loop_id, round=1),
        create_agent_message_event(loop_id, "claude", "Initial proposal", 1),
        create_agent_message_event(loop_id, "gpt4", "Counter-argument", 1),
        create_consensus_event(loop_id),
        create_debate_end_event(loop_id),
    ]


@pytest.fixture
def multiple_mock_ws():
    """Create multiple mock WebSocket connections for broadcast testing."""
    return [
        MockWebSocketConnection(("192.168.1.1", 10001)),
        MockWebSocketConnection(("192.168.1.2", 10002)),
        MockWebSocketConnection(("192.168.1.3", 10003)),
    ]
