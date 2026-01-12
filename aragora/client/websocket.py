"""
WebSocket client for real-time debate streaming.

Provides async iteration over debate events with automatic
reconnection and heartbeat management.

Usage:
    from aragora.client import AragoraClient
    from aragora.client.websocket import DebateStream, stream_debate

    # Class-based API
    stream = DebateStream('ws://localhost:8080', 'debate-123')
    await stream.connect()

    stream.on('agent_message', lambda event: print(event))
    stream.on('consensus', lambda event: print('Consensus!', event))

    # Or async iterator
    async for event in stream_debate('ws://localhost:8080', 'debate-123'):
        print(event.type, event.data)
        if event.type == 'debate_end':
            break
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import AsyncGenerator, Callable, Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Event queue limits
MAX_EVENT_QUEUE_SIZE = 1000  # Prevent unbounded memory growth from event flooding


class DebateEventType(str, Enum):
    """Types of debate events."""
    DEBATE_START = "debate_start"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    AGENT_MESSAGE = "agent_message"
    CRITIQUE = "critique"
    VOTE = "vote"
    CONSENSUS = "consensus"
    DEBATE_END = "debate_end"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"


@dataclass
class DebateEvent:
    """A debate event from WebSocket stream."""
    type: DebateEventType
    debate_id: str
    timestamp: str
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DebateEvent":
        """Create from dictionary."""
        event_type = d.get("type", "error")
        try:
            event_type = DebateEventType(event_type)
        except ValueError:
            event_type = DebateEventType.ERROR

        return cls(
            type=event_type,
            debate_id=d.get("debate_id", ""),
            timestamp=d.get("timestamp", datetime.now().isoformat()),
            data=d.get("data", {}),
        )


@dataclass
class WebSocketOptions:
    """Options for WebSocket connection."""
    reconnect: bool = True
    reconnect_interval: float = 1.0
    max_reconnect_attempts: int = 5
    heartbeat_interval: float = 30.0
    connect_timeout: float = 10.0


EventCallback = Callable[[DebateEvent], None]
ErrorCallback = Callable[[Exception], None]
CloseCallback = Callable[[int, str], None]


class DebateStream:
    """
    WebSocket client for streaming debate events.

    Provides both callback-based and async iterator APIs
    for consuming debate events in real-time.

    Example:
        stream = DebateStream('ws://localhost:8080', 'debate-123')
        await stream.connect()

        # Callback API
        stream.on('agent_message', lambda e: print(e.data))

        # Or iterate
        async for event in stream:
            print(event)
    """

    def __init__(
        self,
        base_url: str,
        debate_id: str,
        options: Optional[WebSocketOptions] = None,
    ):
        """
        Initialize debate stream.

        Args:
            base_url: WebSocket server URL (ws:// or wss://)
            debate_id: ID of debate to stream
            options: Connection options
        """
        self.debate_id = debate_id
        self.options = options or WebSocketOptions()
        self.url = self._build_url(base_url, debate_id)

        self._ws: Optional[Any] = None
        self._reconnect_attempts = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._is_closing = False

        self._event_callbacks: Dict[str, List[EventCallback]] = {}
        self._error_callbacks: List[ErrorCallback] = []
        self._close_callbacks: List[CloseCallback] = []

        self._event_queue: asyncio.Queue[DebateEvent] = asyncio.Queue(maxsize=MAX_EVENT_QUEUE_SIZE)

    def _build_url(self, base_url: str, debate_id: str) -> str:
        """Build WebSocket URL."""
        ws_url = (
            base_url
            .replace("http://", "ws://")
            .replace("https://", "wss://")
            .rstrip("/")
        )
        return f"{ws_url}/ws/debates/{debate_id}"

    async def connect(self) -> None:
        """
        Connect to the WebSocket server.

        Raises:
            ConnectionError: If connection fails
        """
        try:
            import websockets
        except ImportError:
            raise ImportError(
                "websockets package required. Install with: pip install websockets"
            )

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    ping_interval=self.options.heartbeat_interval,
                    ping_timeout=10,
                    close_timeout=5,
                ),
                timeout=self.options.connect_timeout,
            )
            self._reconnect_attempts = 0
            self._is_closing = False

            # Start message receiver
            asyncio.create_task(self._receive_loop())

            logger.debug(f"Connected to {self.url}")

        except asyncio.TimeoutError:
            raise ConnectionError(f"Connection timeout to {self.url}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.url}: {e}")

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        self._is_closing = True

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            self._reconnect_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

    def on(self, event_type: str, callback: EventCallback) -> "DebateStream":
        """
        Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to, or '*' for all events
            callback: Callback function receiving DebateEvent

        Returns:
            Self for chaining
        """
        if event_type not in self._event_callbacks:
            self._event_callbacks[event_type] = []
        self._event_callbacks[event_type].append(callback)
        return self

    def on_error(self, callback: ErrorCallback) -> "DebateStream":
        """Subscribe to errors."""
        self._error_callbacks.append(callback)
        return self

    def on_close(self, callback: CloseCallback) -> "DebateStream":
        """Subscribe to connection close."""
        self._close_callbacks.append(callback)
        return self

    def off(self, event_type: str, callback: EventCallback) -> "DebateStream":
        """Unsubscribe from an event type."""
        if event_type in self._event_callbacks:
            try:
                self._event_callbacks[event_type].remove(callback)
            except ValueError:
                pass
        return self

    async def send(self, data: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if self._ws:
            await self._ws.send(json.dumps(data))

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._ws is not None and self._ws.open

    async def _receive_loop(self) -> None:
        """Receive messages from WebSocket."""
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    event = DebateEvent.from_dict(data)

                    # Emit to callbacks
                    self._emit_event(event)

                    # Add to queue for async iteration (drop if full to prevent blocking)
                    try:
                        self._event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning("Event queue full, dropping event")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message[:100]}")
                except Exception as e:
                    logger.warning(f"Error processing message: {e}")

        except Exception as e:
            if not self._is_closing:
                self._emit_error(e)
                self._emit_close(1006, str(e))

                if self.options.reconnect:
                    await self._attempt_reconnect()

    def _emit_event(self, event: DebateEvent) -> None:
        """Emit event to callbacks."""
        # Type-specific callbacks
        type_key = event.type.value if isinstance(event.type, Enum) else event.type
        if type_key in self._event_callbacks:
            for callback in self._event_callbacks[type_key]:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

        # Wildcard callbacks
        if "*" in self._event_callbacks:
            for callback in self._event_callbacks["*"]:
                try:
                    callback(event)
                except Exception as e:
                    logger.warning(f"Callback error: {e}")

    def _emit_error(self, error: Exception) -> None:
        """Emit error to callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.warning(f"Error callback error: {e}")

    def _emit_close(self, code: int, reason: str) -> None:
        """Emit close to callbacks."""
        for callback in self._close_callbacks:
            try:
                callback(code, reason)
            except Exception as e:
                logger.warning(f"Close callback error: {e}")

    async def _attempt_reconnect(self) -> None:
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_attempts >= self.options.max_reconnect_attempts:
            self._emit_error(ConnectionError("Max reconnect attempts reached"))
            return

        self._reconnect_attempts += 1
        delay = self.options.reconnect_interval * (2 ** (self._reconnect_attempts - 1))

        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect()
        except Exception as e:
            logger.warning(f"Reconnect failed: {e}")
            await self._attempt_reconnect()

    def __aiter__(self):
        """Async iterator support."""
        return self

    async def __anext__(self) -> DebateEvent:
        """Get next event."""
        if self._is_closing and self._event_queue.empty():
            raise StopAsyncIteration

        try:
            event = await asyncio.wait_for(
                self._event_queue.get(),
                timeout=self.options.heartbeat_interval * 2,
            )
            return event
        except asyncio.TimeoutError:
            if not self.is_connected:
                raise StopAsyncIteration
            # Return a synthetic ping event to indicate stream is alive
            return DebateEvent(
                type=DebateEventType.PING,
                debate_id=self.debate_id,
                timestamp=datetime.now().isoformat(),
                data={"keepalive": True},
            )


async def stream_debate(
    base_url: str,
    debate_id: str,
    options: Optional[WebSocketOptions] = None,
) -> AsyncGenerator[DebateEvent, None]:
    """
    Stream debate events as an async generator.

    This is a convenience function that handles connection
    management automatically.

    Args:
        base_url: WebSocket server URL
        debate_id: ID of debate to stream
        options: Connection options

    Yields:
        DebateEvent for each event from the debate

    Example:
        async for event in stream_debate('ws://localhost:8080', 'debate-123'):
            print(event.type, event.data)
            if event.type == DebateEventType.DEBATE_END:
                break
    """
    stream = DebateStream(base_url, debate_id, options)

    try:
        await stream.connect()

        async for event in stream:
            yield event

            if event.type == DebateEventType.DEBATE_END:
                break

    finally:
        await stream.disconnect()


__all__ = [
    "DebateEventType",
    "DebateEvent",
    "WebSocketOptions",
    "DebateStream",
    "stream_debate",
]
