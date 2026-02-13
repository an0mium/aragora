"""WebSocket streaming for the Aragora SDK."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

import websockets

if TYPE_CHECKING:
    try:
        from websockets import ClientConnection as WebSocketClientProtocol
    except Exception:  # pragma: no cover - compatibility fallback
        from websockets.client import WebSocketClientProtocol  # type: ignore

from aragora_client.types import DebateEvent


class DebateStream:
    """
    WebSocket stream for debate events.

    Example:
        >>> stream = DebateStream("ws://localhost:8765", "debate-123")
        >>> stream.on("agent_message", lambda e: print(e.data))
        >>> stream.on("consensus", lambda e: print("Consensus reached!"))
        >>> await stream.connect()
    """

    def __init__(
        self,
        base_url: str,
        debate_id: str,
        *,
        reconnect: bool = True,
        reconnect_interval: float = 1.0,
        max_reconnect_attempts: int = 5,
        heartbeat_interval: float = 30.0,
    ) -> None:
        """
        Initialize the debate stream.

        Args:
            base_url: WebSocket server URL.
            debate_id: ID of the debate to stream.
            reconnect: Whether to auto-reconnect on disconnect.
            reconnect_interval: Base reconnect delay in seconds.
            max_reconnect_attempts: Maximum reconnect attempts.
            heartbeat_interval: Heartbeat ping interval in seconds.
        """
        # Normalize URL
        if base_url.startswith("http://"):
            base_url = base_url.replace("http://", "ws://")
        elif base_url.startswith("https://"):
            base_url = base_url.replace("https://", "wss://")
        elif not base_url.startswith(("ws://", "wss://")):
            base_url = f"ws://{base_url}"

        self.base_url = base_url.rstrip("/")
        self.debate_id = debate_id
        self.reconnect = reconnect
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.heartbeat_interval = heartbeat_interval

        self._ws: WebSocketClientProtocol | None = None
        self._handlers: dict[str, list[Callable[[DebateEvent], None]]] = {}
        self._error_handlers: list[Callable[[Exception], None]] = []
        self._connected = False
        self._should_stop = False
        self._reconnect_attempts = 0

    def on(
        self, event_type: str, handler: Callable[[DebateEvent], None]
    ) -> DebateStream:
        """
        Register an event handler.

        Args:
            event_type: Event type to handle (e.g., "agent_message", "consensus").
            handler: Callback function to handle the event.

        Returns:
            Self for chaining.
        """
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        return self

    def on_error(self, handler: Callable[[Exception], None]) -> DebateStream:
        """
        Register an error handler.

        Args:
            handler: Callback function to handle errors.

        Returns:
            Self for chaining.
        """
        self._error_handlers.append(handler)
        return self

    async def connect(self) -> None:
        """Connect to the WebSocket and start receiving events."""
        self._should_stop = False
        ws_url = f"{self.base_url}/ws/debates/{self.debate_id}"

        while not self._should_stop:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_attempts = 0

                    # Start heartbeat
                    heartbeat_task = asyncio.create_task(self._heartbeat())

                    try:
                        async for message in ws:
                            if self._should_stop:
                                break
                            await self._handle_message(message)
                    finally:
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass

            except websockets.ConnectionClosed:
                self._connected = False
                if not self.reconnect or self._should_stop:
                    break
                if self._reconnect_attempts >= self.max_reconnect_attempts:
                    max_attempts = self.max_reconnect_attempts
                    self._emit_error(
                        Exception(f"Max reconnect attempts ({max_attempts}) exceeded")
                    )
                    break
                self._reconnect_attempts += 1
                delay = self.reconnect_interval * (2 ** (self._reconnect_attempts - 1))
                await asyncio.sleep(delay)

            except Exception as e:
                self._connected = False
                self._emit_error(e)
                if not self.reconnect or self._should_stop:
                    break
                self._reconnect_attempts += 1
                if self._reconnect_attempts >= self.max_reconnect_attempts:
                    break
                delay = self.reconnect_interval * (2 ** (self._reconnect_attempts - 1))
                await asyncio.sleep(delay)

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket."""
        self._should_stop = True
        self._connected = False
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def _heartbeat(self) -> None:
        """Send periodic heartbeat pings."""
        while self._connected and self._ws:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if self._ws and self._connected:
                    await self._ws.ping()
            except Exception:
                break

    async def _handle_message(self, message: str | bytes) -> None:
        """Handle an incoming WebSocket message."""
        try:
            if isinstance(message, bytes):
                message = message.decode("utf-8")

            data = json.loads(message)
            event = DebateEvent(
                type=data.get("type", "unknown"),
                data=data.get("data", {}),
                loop_id=data.get("loop_id") or data.get("data", {}).get("debate_id"),
            )

            # Call handlers for this event type
            handlers = self._handlers.get(event.type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    self._emit_error(e)

            # Call handlers for "*" (all events)
            for handler in self._handlers.get("*", []):
                try:
                    handler(event)
                except Exception as e:
                    self._emit_error(e)

        except json.JSONDecodeError as e:
            self._emit_error(e)

    def _emit_error(self, error: Exception) -> None:
        """Emit an error to all error handlers."""
        for handler in self._error_handlers:
            try:
                handler(error)
            except Exception:
                pass  # Avoid infinite loops

    @property
    def connected(self) -> bool:
        """Whether the stream is currently connected."""
        return self._connected


async def stream_debate(
    base_url: str,
    debate_id: str,
    *,
    reconnect: bool = True,
) -> AsyncIterator[DebateEvent]:
    """
    Stream debate events as an async iterator.

    Example:
        >>> async for event in stream_debate("ws://localhost:8765", "debate-123"):
        ...     print(event.type, event.data)
        ...     if event.type == "debate_end":
        ...         break

    Args:
        base_url: WebSocket server URL.
        debate_id: ID of the debate to stream.
        reconnect: Whether to auto-reconnect on disconnect.

    Yields:
        DebateEvent objects.
    """
    # Normalize URL
    if base_url.startswith("http://"):
        base_url = base_url.replace("http://", "ws://")
    elif base_url.startswith("https://"):
        base_url = base_url.replace("https://", "wss://")
    elif not base_url.startswith(("ws://", "wss://")):
        base_url = f"ws://{base_url}"

    ws_url = f"{base_url.rstrip('/')}/ws/debates/{debate_id}"

    async with websockets.connect(ws_url) as ws:
        async for message in ws:
            if isinstance(message, bytes):
                message = message.decode("utf-8")

            try:
                data = json.loads(message)
                yield DebateEvent(
                    type=data.get("type", "unknown"),
                    data=data.get("data", {}),
                    loop_id=data.get("loop_id")
                    or data.get("data", {}).get("debate_id"),
                )
            except json.JSONDecodeError:
                continue
