"""
Aragora WebSocket Client

Provides real-time streaming for debate events, agent messages, and consensus.

Usage::

    from sdk.python.aragora.websocket import AragoraWebSocket, WebSocketOptions

    ws = AragoraWebSocket("http://localhost:8080", api_key="your-key")
    ws.on("agent_message", lambda event: print(event))
    await ws.connect(debate_id="debate-123")
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event types supported by the Aragora WebSocket protocol
# ---------------------------------------------------------------------------

EVENT_TYPES: tuple[str, ...] = (
    "connected",
    "disconnected",
    "error",
    "debate_start",
    "round_start",
    "agent_message",
    "propose",
    "critique",
    "revision",
    "synthesis",
    "vote",
    "consensus",
    "consensus_reached",
    "debate_end",
    "phase_change",
    "audience_suggestion",
    "user_vote",
    "warning",
    "message",
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WebSocketOptions:
    """Configuration options for the WebSocket connection."""

    auto_reconnect: bool = True
    max_reconnect_attempts: int = 5
    reconnect_delay: float = 1.0
    heartbeat_interval: float = 30.0


@dataclass
class WebSocketEvent:
    """A single event received over the WebSocket connection.

    Attributes:
        type: Event type string (e.g. ``"debate_start"``, ``"agent_message"``).
        data: Raw event payload as a dictionary.
        timestamp: ISO timestamp of when the event occurred.
        debate_id: ID of the debate this event belongs to.
        typed_data: Parsed typed event object, if available. Check this for
            type-safe access to event fields instead of using ``data``.
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    debate_id: str | None = None
    typed_data: Any = None


# ---------------------------------------------------------------------------
# WebSocket state constants
# ---------------------------------------------------------------------------

_STATE_CONNECTING = "connecting"
_STATE_CONNECTED = "connected"
_STATE_DISCONNECTED = "disconnected"
_STATE_RECONNECTING = "reconnecting"


# ---------------------------------------------------------------------------
# AragoraWebSocket
# ---------------------------------------------------------------------------


class AragoraWebSocket:
    """Real-time WebSocket client for streaming Aragora debate events.

    Parameters:
        base_url: The HTTP base URL of the Aragora server (e.g. ``http://localhost:8080``).
        api_key:  Optional API key used for authentication.
        ws_url:   Explicit WebSocket URL.  When *None*, derived from *base_url*.
        options:  Connection tuning knobs (reconnect behaviour, heartbeat, etc.).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        ws_url: str | None = None,
        options: WebSocketOptions | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.ws_url = ws_url
        self.options = options or WebSocketOptions()

        # Internal state
        self._state: str = _STATE_DISCONNECTED
        self._ws: Any = None  # websockets connection object
        self._reconnect_attempts: int = 0
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._closing: bool = False

        # Event handler registry: event_type -> list[handler]
        self._handlers: dict[str, list[Callable[..., Any]]] = {evt: [] for evt in EVENT_TYPES}

        # Queue used for the async-iterator interface (``events()``).
        # We use ``queue.Queue`` (thread-safe, sync put) so that
        # ``_handle_message`` can enqueue without needing ``await``.
        self._event_queue: queue.Queue[WebSocketEvent | None] = queue.Queue()

        # Debate subscriptions tracked locally
        self._subscriptions: set[str] = set()

    # -- public properties ---------------------------------------------------

    @property
    def state(self) -> str:
        """Return the current connection state string."""
        return self._state

    # -- connection lifecycle ------------------------------------------------

    async def connect(self, debate_id: str | None = None) -> None:
        """Open a WebSocket connection to the server.

        If *debate_id* is given it is included as a query parameter so that the
        server immediately scopes the stream to that debate.
        """
        if self._state in (_STATE_CONNECTED, _STATE_CONNECTING):
            return

        self._state = _STATE_CONNECTING
        self._closing = False
        url = self._build_ws_url(debate_id)

        try:
            # Import websockets lazily so the module can be loaded even when
            # the optional dependency is not installed.
            import websockets  # type: ignore[import-untyped]

            self._ws = await websockets.connect(url)  # type: ignore[attr-defined]
            self._state = _STATE_CONNECTED
            self._reconnect_attempts = 0

            # Start background tasks
            self._receive_task = asyncio.ensure_future(self._receive_loop())
            self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())

            self._emit("connected", {})
        except Exception as exc:
            self._state = _STATE_DISCONNECTED
            self._emit("error", {"error": str(exc)})
            raise

    async def close(self) -> None:
        """Gracefully close the WebSocket connection."""
        self._closing = True
        self.options.auto_reconnect = False
        self._cleanup()

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        self._state = _STATE_DISCONNECTED

        # Signal the async-iterator that we are done.
        self._event_queue.put(None)

    # -- subscriptions -------------------------------------------------------

    def subscribe(self, debate_id: str) -> None:
        """Subscribe to events for a specific debate."""
        self._subscriptions.add(debate_id)
        self._send({"type": "subscribe", "debate_id": debate_id})

    def unsubscribe(self, debate_id: str) -> None:
        """Unsubscribe from a debate's event stream."""
        self._subscriptions.discard(debate_id)
        self._send({"type": "unsubscribe", "debate_id": debate_id})

    # -- event handler registration ------------------------------------------

    def on(self, event: str, handler: Callable[..., Any]) -> Callable[[], None]:
        """Register *handler* to be called whenever *event* fires.

        Returns a callable that, when invoked, removes the handler (i.e. an
        *unsubscribe* function).
        """
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)

        def _unsubscribe() -> None:
            self.off(event, handler)

        return _unsubscribe

    def off(self, event: str, handler: Callable[..., Any]) -> None:
        """Remove a previously registered handler for *event*."""
        handlers = self._handlers.get(event)
        if handlers is not None:
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    # -- async iterator interface --------------------------------------------

    async def events(self) -> AsyncIterator[WebSocketEvent]:
        """Yield events as an async iterator.

        The iterator terminates when the connection is closed.

        Example::

            async for event in ws.events():
                print(event.type, event.data)
        """
        loop = asyncio.get_event_loop()
        while True:
            # Retrieve from the sync queue without blocking the event loop.
            event = await loop.run_in_executor(None, self._event_queue.get)
            if event is None:
                # Sentinel: stream finished.
                break
            yield event

    # -- internals -----------------------------------------------------------

    def _build_ws_url(self, debate_id: str | None = None) -> str:
        """Construct the full WebSocket URL including auth and debate params."""
        ws_url = self.ws_url

        if not ws_url:
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")

        # Ensure the path ends with /ws
        if not ws_url.endswith("/ws"):
            ws_url = ws_url.rstrip("/") + "/ws"

        # Build query string
        params: list[str] = []
        if debate_id:
            params.append(f"debate_id={quote(debate_id, safe='')}")
        if self.api_key:
            params.append(f"token={quote(self.api_key, safe='')}")

        if params:
            ws_url += "?" + "&".join(params)

        return ws_url

    def _handle_message(self, raw: str) -> None:
        """Parse and dispatch an incoming message."""
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self._emit("error", {"error": f"Failed to parse message: {raw}"})
            return

        event = WebSocketEvent(
            type=payload.get("type", "message"),
            data=payload.get("data", {}),
            timestamp=payload.get("timestamp", ""),
            debate_id=payload.get("debate_id"),
        )

        # Parse typed event data if a matching event class exists
        from .events import EVENT_CLASS_MAP

        cls = EVENT_CLASS_MAP.get(event.type)
        if cls:
            try:
                # Only pass fields that exist in the dataclass
                event.typed_data = cls(
                    **{k: v for k, v in event.data.items() if k in cls.__dataclass_fields__}
                )
            except (TypeError, KeyError):
                pass  # Fall back to untyped data dict

        # Emit to the generic ``message`` handlers.
        self._emit("message", event)

        # Emit to specific event-type handlers.
        if event.type in self._handlers:
            self._emit(event.type, event)

        # Also enqueue for the async iterator.
        self._event_queue.put(event)

    def _emit(self, event: str, data: Any) -> None:
        """Invoke all registered handlers for *event*."""
        for handler in list(self._handlers.get(event, [])):
            try:
                handler(data)
            except Exception:
                logger.exception("Error in %s handler", event)

    def _send(self, data: dict[str, Any]) -> None:
        """Queue a JSON message for sending (fire-and-forget)."""
        if self._ws is not None and self._state == _STATE_CONNECTED:
            try:
                asyncio.ensure_future(self._ws.send(json.dumps(data)))
            except Exception:
                logger.debug("Failed to send message", exc_info=True)

    # -- background loops ----------------------------------------------------

    async def _receive_loop(self) -> None:
        """Continuously read messages from the WebSocket."""
        try:
            async for raw_message in self._ws:
                self._handle_message(
                    raw_message if isinstance(raw_message, str) else raw_message.decode()
                )
        except Exception as exc:
            if not self._closing:
                self._emit("error", {"error": str(exc)})
                self._handle_disconnect(1006, str(exc))

    async def _heartbeat_loop(self) -> None:
        """Periodically send a ping frame to keep the connection alive."""
        try:
            while self._state == _STATE_CONNECTED:
                await asyncio.sleep(self.options.heartbeat_interval)
                if self._state == _STATE_CONNECTED:
                    self._send({"type": "ping"})
        except asyncio.CancelledError:
            pass

    # -- reconnection --------------------------------------------------------

    def _handle_disconnect(self, code: int, reason: str) -> None:
        """Handle an unexpected disconnect."""
        self._cleanup()
        self._state = _STATE_DISCONNECTED
        self._emit("disconnected", {"code": code, "reason": reason})

        if (
            self.options.auto_reconnect
            and self._reconnect_attempts < self.options.max_reconnect_attempts
        ):
            self._schedule_reconnect()

    def _schedule_reconnect(self) -> None:
        """Schedule a reconnection with exponential back-off."""
        self._state = _STATE_RECONNECTING
        delay = self.options.reconnect_delay * (2**self._reconnect_attempts)
        self._reconnect_attempts += 1

        async def _do_reconnect() -> None:
            await asyncio.sleep(delay)
            try:
                await self.connect()
            except Exception:
                pass  # connect() will emit errors and trigger disconnect

        self._reconnect_task = asyncio.ensure_future(_do_reconnect())

    def _cleanup(self) -> None:
        """Cancel background tasks."""
        for task in (self._heartbeat_task, self._receive_task, self._reconnect_task):
            if task is not None and not task.done():
                task.cancel()
        self._heartbeat_task = None
        self._receive_task = None
        self._reconnect_task = None


# ---------------------------------------------------------------------------
# Convenience: async generator for streaming debate events
# ---------------------------------------------------------------------------


async def stream_debate(
    base_url: str,
    debate_id: str | None = None,
    api_key: str | None = None,
    options: WebSocketOptions | None = None,
) -> AsyncIterator[WebSocketEvent]:
    """High-level async generator that connects, streams, and cleans up.

    Example::

        async for event in stream_debate("http://localhost:8080", debate_id="d1"):
            print(event.type, event.data)
            if event.type == "debate_end":
                break
    """
    ws = AragoraWebSocket(base_url, api_key=api_key, options=options)
    await ws.connect(debate_id=debate_id)

    try:
        async for event in ws.events():
            yield event
            if event.type in ("debate_end", "error"):
                break
    finally:
        await ws.close()
