"""
Aragora WebSocket Streaming Client.

Provides real-time streaming for debate events, agent messages, and consensus.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
)

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.client import WebSocketClientProtocol

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    WebSocketClientProtocol = Any  # type: ignore


class WebSocketState(str, Enum):
    """WebSocket connection state."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"


@dataclass
class WebSocketEvent:
    """A WebSocket event from the Aragora server."""

    type: str
    data: Optional[Dict[str, Any]] = None
    debate_id: Optional[str] = None
    timestamp: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WebSocketEvent":
        """Create from dictionary."""
        return cls(
            type=d.get("type", "unknown"),
            data=d.get("data"),
            debate_id=d.get("debate_id"),
            timestamp=d.get("timestamp"),
        )


# ============================================================================
# Typed WebSocket Events (discriminated unions for type safety)
# ============================================================================


@dataclass
class DebateStartEvent:
    """Event fired when a debate begins."""

    debate_id: str
    task: str = ""
    agents: List[str] = None  # type: ignore[assignment]
    timestamp: Optional[str] = None
    type: str = "debate_start"

    def __post_init__(self):
        if self.agents is None:
            self.agents = []


@dataclass
class RoundStartEvent:
    """Event fired when a new round begins."""

    debate_id: str
    round_number: int = 0
    phase: str = ""
    timestamp: Optional[str] = None
    type: str = "round_start"


@dataclass
class RoundEndEvent:
    """Event fired when a round ends."""

    debate_id: str
    round_number: int = 0
    timestamp: Optional[str] = None
    type: str = "round_end"


@dataclass
class AgentMessageEvent:
    """Event fired when an agent sends a message."""

    debate_id: str
    agent: str = ""
    content: str = ""
    round_number: int = 0
    role: str = ""
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    type: str = "agent_message"


@dataclass
class ProposeEvent:
    """Event fired when an agent proposes an answer."""

    debate_id: str
    agent: str = ""
    proposal: str = ""
    round_number: int = 0
    confidence: Optional[float] = None
    timestamp: Optional[str] = None
    type: str = "propose"


@dataclass
class CritiqueEvent:
    """Event fired when an agent critiques a proposal."""

    debate_id: str
    agent: str = ""
    target_agent: str = ""
    critique: str = ""
    round_number: int = 0
    timestamp: Optional[str] = None
    type: str = "critique"


@dataclass
class RevisionEvent:
    """Event fired when an agent revises their proposal."""

    debate_id: str
    agent: str = ""
    revision: str = ""
    round_number: int = 0
    timestamp: Optional[str] = None
    type: str = "revision"


@dataclass
class VoteEvent:
    """Event fired when an agent casts a vote."""

    debate_id: str
    agent: str = ""
    choice: str = ""
    confidence: Optional[float] = None
    round_number: int = 0
    timestamp: Optional[str] = None
    type: str = "vote"


@dataclass
class ConsensusEvent:
    """Event fired when consensus status is updated."""

    debate_id: str
    reached: bool = False
    agreement: Optional[float] = None
    final_answer: Optional[str] = None
    timestamp: Optional[str] = None
    type: str = "consensus"


@dataclass
class DebateEndEvent:
    """Event fired when a debate concludes."""

    debate_id: str
    final_answer: Optional[str] = None
    consensus_reached: bool = False
    total_rounds: int = 0
    timestamp: Optional[str] = None
    type: str = "debate_end"


@dataclass
class PhaseChangeEvent:
    """Event fired when the debate phase changes."""

    debate_id: str
    phase: str = ""
    previous_phase: Optional[str] = None
    timestamp: Optional[str] = None
    type: str = "phase_change"


@dataclass
class ErrorEvent:
    """Event fired when an error occurs."""

    message: str = ""
    code: Optional[str] = None
    debate_id: Optional[str] = None
    timestamp: Optional[str] = None
    type: str = "error"


@dataclass
class HeartbeatEvent:
    """Event fired for connection keepalive."""

    timestamp: Optional[str] = None
    type: str = "heartbeat"


# Union type for all typed events
TypedWebSocketEvent = (
    DebateStartEvent
    | RoundStartEvent
    | RoundEndEvent
    | AgentMessageEvent
    | ProposeEvent
    | CritiqueEvent
    | RevisionEvent
    | VoteEvent
    | ConsensusEvent
    | DebateEndEvent
    | PhaseChangeEvent
    | ErrorEvent
    | HeartbeatEvent
    | WebSocketEvent  # Fallback for unknown events
)


def parse_typed_event(data: Dict[str, Any]) -> TypedWebSocketEvent:
    """Parse raw event data into a typed event object.

    Args:
        data: Raw event dictionary from WebSocket

    Returns:
        A typed event object based on the event type

    Example:
        ```python
        event = parse_typed_event({"type": "agent_message", "agent": "claude", ...})
        if isinstance(event, AgentMessageEvent):
            print(f"{event.agent}: {event.content}")
        ```
    """
    event_type = data.get("type", "")
    event_data = data.get("data", {})
    debate_id = data.get("debate_id", "")
    timestamp = data.get("timestamp")

    if event_type == "debate_start":
        return DebateStartEvent(
            debate_id=debate_id,
            task=event_data.get("task", ""),
            agents=event_data.get("agents", []),
            timestamp=timestamp,
        )

    if event_type == "round_start":
        return RoundStartEvent(
            debate_id=debate_id,
            round_number=event_data.get("round_number", 0),
            phase=event_data.get("phase", ""),
            timestamp=timestamp,
        )

    if event_type == "round_end":
        return RoundEndEvent(
            debate_id=debate_id,
            round_number=event_data.get("round_number", 0),
            timestamp=timestamp,
        )

    if event_type == "agent_message":
        return AgentMessageEvent(
            debate_id=debate_id,
            agent=event_data.get("agent", ""),
            content=event_data.get("content", ""),
            round_number=event_data.get("round_number", 0),
            role=event_data.get("role", ""),
            confidence=event_data.get("confidence"),
            timestamp=timestamp,
        )

    if event_type == "propose":
        return ProposeEvent(
            debate_id=debate_id,
            agent=event_data.get("agent", ""),
            proposal=event_data.get("proposal", ""),
            round_number=event_data.get("round_number", 0),
            confidence=event_data.get("confidence"),
            timestamp=timestamp,
        )

    if event_type == "critique":
        return CritiqueEvent(
            debate_id=debate_id,
            agent=event_data.get("agent", ""),
            target_agent=event_data.get("target_agent", ""),
            critique=event_data.get("critique", ""),
            round_number=event_data.get("round_number", 0),
            timestamp=timestamp,
        )

    if event_type == "revision":
        return RevisionEvent(
            debate_id=debate_id,
            agent=event_data.get("agent", ""),
            revision=event_data.get("revision", ""),
            round_number=event_data.get("round_number", 0),
            timestamp=timestamp,
        )

    if event_type == "vote":
        return VoteEvent(
            debate_id=debate_id,
            agent=event_data.get("agent", ""),
            choice=event_data.get("choice", ""),
            confidence=event_data.get("confidence"),
            round_number=event_data.get("round_number", 0),
            timestamp=timestamp,
        )

    if event_type in ("consensus", "consensus_reached"):
        return ConsensusEvent(
            debate_id=debate_id,
            reached=event_data.get("reached", False),
            agreement=event_data.get("agreement"),
            final_answer=event_data.get("final_answer"),
            timestamp=timestamp,
        )

    if event_type == "debate_end":
        return DebateEndEvent(
            debate_id=debate_id,
            final_answer=event_data.get("final_answer"),
            consensus_reached=event_data.get("consensus_reached", False),
            total_rounds=event_data.get("total_rounds", 0),
            timestamp=timestamp,
        )

    if event_type == "phase_change":
        return PhaseChangeEvent(
            debate_id=debate_id,
            phase=event_data.get("phase", ""),
            previous_phase=event_data.get("previous_phase"),
            timestamp=timestamp,
        )

    if event_type == "error":
        return ErrorEvent(
            message=event_data.get("message", ""),
            code=event_data.get("code"),
            debate_id=debate_id,
            timestamp=timestamp,
        )

    if event_type == "heartbeat":
        return HeartbeatEvent(timestamp=timestamp)

    # Fallback to generic event for unknown types
    return WebSocketEvent.from_dict(data)


@dataclass
class WebSocketOptions:
    """Options for WebSocket connection."""

    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    heartbeat_interval: float = 30.0


EventHandler = Callable[[Any], None]
T = TypeVar("T")


class AragoraWebSocket:
    """
    WebSocket client for streaming Aragora debate events.

    Provides:
    - Real-time event streaming
    - Automatic reconnection
    - Event filtering by debate ID
    - Async iterator support

    Example:
        ```python
        from aragora import AragoraWebSocket

        ws = AragoraWebSocket(base_url="http://localhost:8080")
        await ws.connect()

        @ws.on("agent_message")
        def handle_message(event):
            print(f"{event['agent']}: {event['content']}")

        # Or use async iteration:
        async for event in ws.stream_events():
            print(event)
        ```
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        ws_url: Optional[str] = None,
        options: Optional[WebSocketOptions] = None,
    ):
        if not HAS_WEBSOCKETS:
            raise ImportError(
                "websockets is required for streaming. Install with: pip install websockets"
            )

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._ws_url = ws_url
        self.options = options or WebSocketOptions()
        self._ws: Optional[WebSocketClientProtocol] = None
        self._state = WebSocketState.DISCONNECTED
        self._reconnect_attempts = 0
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._heartbeat_task: Optional[asyncio.Task[None]] = None
        self._receive_task: Optional[asyncio.Task[None]] = None

    @property
    def state(self) -> WebSocketState:
        """Get current connection state."""
        return self._state

    def _build_ws_url(self, debate_id: Optional[str] = None) -> str:
        """Build WebSocket URL from config."""
        if self._ws_url:
            ws_url = self._ws_url
        else:
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")

        if not ws_url.endswith("/ws"):
            ws_url = ws_url.rstrip("/") + "/ws"

        params = []
        if debate_id:
            params.append(f"debate_id={debate_id}")
        if self.api_key:
            params.append(f"token={self.api_key}")

        if params:
            ws_url += "?" + "&".join(params)

        return ws_url

    async def connect(self, debate_id: Optional[str] = None) -> None:
        """
        Connect to the WebSocket server.

        Args:
            debate_id: Optional debate ID to subscribe to immediately.
        """
        if self._state in (WebSocketState.CONNECTED, WebSocketState.CONNECTING):
            return

        self._state = WebSocketState.CONNECTING
        ws_url = self._build_ws_url(debate_id)

        try:
            self._ws = await websockets.connect(ws_url)
            self._state = WebSocketState.CONNECTED
            self._reconnect_attempts = 0

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Emit connected event
            self._emit("connected", None)

            logger.info(f"Connected to WebSocket: {ws_url}")
        except Exception as e:
            self._state = WebSocketState.DISCONNECTED
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        self.options.auto_reconnect = False
        await self._cleanup()

        if self._ws:
            await self._ws.close(1000, "Client disconnect")
            self._ws = None

        self._state = WebSocketState.DISCONNECTED
        self._emit("disconnected", {"code": 1000, "reason": "Client disconnect"})

    async def _cleanup(self) -> None:
        """Clean up tasks."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        try:
            while self._state == WebSocketState.CONNECTED:
                await asyncio.sleep(self.options.heartbeat_interval)
                await self.send({"type": "ping"})
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Heartbeat error: {e}")

    async def send(self, data: Dict[str, Any]) -> None:
        """Send a message to the server."""
        if self._ws and self._state == WebSocketState.CONNECTED:
            await self._ws.send(json.dumps(data))

    async def subscribe(self, debate_id: str) -> None:
        """Subscribe to a specific debate's events."""
        await self.send({"type": "subscribe", "debate_id": debate_id})

    async def unsubscribe(self, debate_id: str) -> None:
        """Unsubscribe from a debate's events."""
        await self.send({"type": "unsubscribe", "debate_id": debate_id})

    def on(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        """
        Register an event handler.

        Args:
            event_type: Event type to handle (e.g., "agent_message", "debate_end")
            handler: Callback function

        Returns:
            Unsubscribe function
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

        return unsubscribe

    def off(self, event_type: str, handler: EventHandler) -> None:
        """Remove an event handler."""
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def once(
        self,
        event_type: str,
        timeout: Optional[float] = None,
    ) -> WebSocketEvent:
        """
        Wait for a single event of the specified type.

        This is useful for waiting on specific events like 'debate_start' or
        'consensus_reached' without setting up persistent handlers.

        Args:
            event_type: The event type to wait for (e.g., 'debate_start', 'consensus')
            timeout: Optional timeout in seconds. If exceeded, raises asyncio.TimeoutError.

        Returns:
            The first WebSocketEvent matching the specified type.

        Raises:
            asyncio.TimeoutError: If timeout is specified and exceeded.

        Example:
            ```python
            ws = AragoraWebSocket(base_url="http://localhost:8080")
            await ws.connect(debate_id="debate-123")

            # Wait for debate to start (with 30s timeout)
            start_event = await ws.once("debate_start", timeout=30.0)
            print(f"Debate started: {start_event.data}")

            # Wait for consensus (no timeout)
            consensus = await ws.once("consensus_reached")
            print(f"Consensus: {consensus.data}")
            ```
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[WebSocketEvent] = loop.create_future()

        def handler(data: Any) -> None:
            if not future.done():
                event = WebSocketEvent(type=event_type, data=data)
                future.set_result(event)

        # Register the one-time handler
        self.on(event_type, handler)

        try:
            if timeout is not None:
                return await asyncio.wait_for(future, timeout)
            return await future
        finally:
            # Always clean up the handler
            self.off(event_type, handler)

    def _emit(self, event_type: str, data: Any) -> None:
        """Emit an event to handlers."""
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in {event_type} handler: {e}")

        # Also emit to "message" handlers for all events
        if event_type not in ("message", "connected", "disconnected", "error"):
            for handler in self._handlers.get("message", []):
                try:
                    handler(WebSocketEvent(type=event_type, data=data))
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")

    async def stream_events(
        self,
        debate_id: Optional[str] = None,
    ) -> AsyncGenerator[WebSocketEvent, None]:
        """
        Stream events as an async generator.

        Args:
            debate_id: Optional debate ID to filter events.

        Yields:
            WebSocketEvent objects

        Example:
            ```python
            async for event in ws.stream_events(debate_id="debate-123"):
                if event.type == "agent_message":
                    print(event.data)
                elif event.type == "debate_end":
                    break
            ```
        """
        if self._state != WebSocketState.CONNECTED:
            await self.connect(debate_id)

        try:
            while self._state == WebSocketState.CONNECTED and self._ws:
                try:
                    message = await self._ws.recv()
                    event_data = json.loads(message)
                    event = WebSocketEvent.from_dict(event_data)

                    # Filter by debate ID if specified
                    if debate_id:
                        event_debate_id = (
                            event.debate_id or event.data.get("debate_id") if event.data else None
                        )
                        if event_debate_id and event_debate_id != debate_id:
                            continue

                    # Emit to handlers
                    self._emit(event.type, event.data)

                    yield event

                    # Check for terminal events
                    if event.type in ("debate_end", "error"):
                        break

                except websockets.ConnectionClosed as e:
                    logger.info(f"WebSocket closed: {e.code} {e.reason}")
                    self._state = WebSocketState.DISCONNECTED
                    self._emit("disconnected", {"code": e.code, "reason": e.reason})

                    if (
                        self.options.auto_reconnect
                        and self._reconnect_attempts < self.options.max_reconnect_attempts
                    ):
                        await self._reconnect(debate_id)
                    else:
                        break

        finally:
            await self._cleanup()

    async def _reconnect(self, debate_id: Optional[str] = None) -> None:
        """Attempt to reconnect."""
        self._state = WebSocketState.RECONNECTING
        delay = self.options.reconnect_delay * (2**self._reconnect_attempts)
        self._reconnect_attempts += 1

        logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_attempts})")
        await asyncio.sleep(delay)

        try:
            await self.connect(debate_id)
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")


async def stream_debate(
    base_url: str,
    debate_id: Optional[str] = None,
    api_key: Optional[str] = None,
    options: Optional[WebSocketOptions] = None,
) -> AsyncGenerator[WebSocketEvent, None]:
    """
    Stream debate events.

    Convenience function that creates a WebSocket client and streams events.

    Args:
        base_url: Aragora server URL
        debate_id: Optional debate ID to filter events
        api_key: Optional API key for authentication
        options: WebSocket options

    Yields:
        WebSocketEvent objects

    Example:
        ```python
        from aragora.streaming import stream_debate

        async for event in stream_debate(
            "http://localhost:8080",
            debate_id="debate-123",
        ):
            print(f"{event.type}: {event.data}")
        ```
    """
    ws = AragoraWebSocket(
        base_url=base_url,
        api_key=api_key,
        options=options,
    )

    try:
        async for event in ws.stream_events(debate_id):
            yield event
    finally:
        await ws.disconnect()


async def stream_debate_by_id(
    base_url: str,
    debate_id: str,
    api_key: Optional[str] = None,
    options: Optional[WebSocketOptions] = None,
) -> AsyncGenerator[WebSocketEvent, None]:
    """
    Stream events for a specific debate by ID.

    Convenience wrapper around stream_debate() that requires a debate_id.
    Use this when you always want to stream a specific debate's events.

    Args:
        base_url: Aragora server URL
        debate_id: The debate ID to stream (required)
        api_key: Optional API key for authentication
        options: WebSocket options

    Yields:
        WebSocketEvent objects for the specified debate

    Example:
        ```python
        from aragora.streaming import stream_debate_by_id

        async for event in stream_debate_by_id(
            "http://localhost:8080",
            debate_id="debate-123",
        ):
            print(f"{event.type}: {event.data}")
        ```
    """
    async for event in stream_debate(
        base_url=base_url,
        debate_id=debate_id,
        api_key=api_key,
        options=options,
    ):
        yield event
