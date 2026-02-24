"""
Streaming connection reliability and reconnection hardening.

Provides automatic reconnection with exponential backoff, connection state
machine, health check ping/pong, and message buffering during reconnection
for WebSocket and enterprise streaming connections.

Integrates with the existing CircuitBreaker from aragora.resilience.

Usage:
    from aragora.streaming.reliability import (
        ReconnectPolicy,
        ReliableWebSocket,
        ReliableKafkaConsumer,
    )

    # WebSocket with automatic reconnection
    policy = ReconnectPolicy(max_retries=10, base_delay=1.0)
    ws = ReliableWebSocket("ws://localhost:8765", policy=policy)
    await ws.connect()

    # Kafka consumer with reliability wrapper
    consumer = ReliableKafkaConsumer(
        bootstrap_servers="localhost:9092",
        topics=["events"],
        policy=ReconnectPolicy(max_retries=5),
    )
    await consumer.connect()
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
from collections.abc import Awaitable, Callable

if TYPE_CHECKING:
    from aragora.resilience.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection state machine
# ---------------------------------------------------------------------------


class ConnectionState(str, Enum):
    """States for a reliable connection."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DISCONNECTING = "disconnecting"


# ---------------------------------------------------------------------------
# Reconnect policy
# ---------------------------------------------------------------------------


@dataclass
class ReconnectPolicy:
    """Configuration for reconnection behaviour.

    Attributes:
        max_retries: Maximum number of reconnection attempts before giving up.
            Set to 0 to disable automatic reconnection.
        base_delay: Initial delay in seconds before the first reconnection attempt.
        max_delay: Upper-bound cap on the computed backoff delay.
        backoff_factor: Multiplier applied per attempt (delay = base_delay * factor^attempt).
        jitter: Whether to add random jitter to prevent thundering herd.
    """

    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Return the backoff delay for the given attempt number (0-indexed).

        Uses exponential backoff with optional jitter:
            delay = min(base_delay * backoff_factor^attempt, max_delay)
            if jitter: delay *= uniform(0.5, 1.0)
        """
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay *= random.uniform(0.5, 1.0)
        return max(0.0, delay)


# ---------------------------------------------------------------------------
# Reliable connection base class
# ---------------------------------------------------------------------------


class ReliableConnection:
    """Base class for connections with automatic reconnection.

    Provides:
    - Exponential backoff with jitter for reconnection
    - Connection state machine (connecting, connected, disconnecting,
      disconnected, reconnecting)
    - Health check ping/pong with configurable interval
    - Message buffer during reconnection (bounded deque)
    - Event hooks: on_connect, on_disconnect, on_reconnect, on_message_dropped
    - Optional CircuitBreaker integration from aragora.resilience

    Subclasses must implement ``_do_connect``, ``_do_disconnect``,
    and ``_do_health_check``.
    """

    def __init__(
        self,
        *,
        policy: ReconnectPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        buffer_size: int = 1000,
        health_check_interval: float = 30.0,
        on_connect: Callable[[], Awaitable[None] | None] | None = None,
        on_disconnect: Callable[[Exception | None], Awaitable[None] | None] | None = None,
        on_reconnect: Callable[[int], Awaitable[None] | None] | None = None,
        on_message_dropped: Callable[[Any], Awaitable[None] | None] | None = None,
    ) -> None:
        self._policy = policy or ReconnectPolicy()
        self._circuit_breaker = circuit_breaker
        self._state = ConnectionState.DISCONNECTED
        self._buffer: deque[Any] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size
        self._health_check_interval = health_check_interval

        # Event hooks
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        self._on_reconnect = on_reconnect
        self._on_message_dropped = on_message_dropped

        # Internal bookkeeping
        self._reconnect_attempts = 0
        self._health_task: asyncio.Task[None] | None = None
        self._messages_dropped = 0
        self._last_connected_at: float | None = None

    # -- public properties --------------------------------------------------

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    @property
    def buffered_count(self) -> int:
        """Number of messages currently buffered."""
        return len(self._buffer)

    @property
    def messages_dropped(self) -> int:
        """Total messages dropped because the buffer was full."""
        return self._messages_dropped

    # -- public API ---------------------------------------------------------

    async def connect(self) -> bool:
        """Establish the connection.

        Returns True if the connection was established successfully.
        """
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            return self._state == ConnectionState.CONNECTED

        self._set_state(ConnectionState.CONNECTING)

        # Check circuit breaker before attempting
        if self._circuit_breaker and not self._circuit_breaker.can_proceed():
            logger.warning("[ReliableConnection] Circuit breaker is open, aborting connect")
            self._set_state(ConnectionState.DISCONNECTED)
            return False

        try:
            await self._do_connect()
            self._set_state(ConnectionState.CONNECTED)
            self._reconnect_attempts = 0
            self._last_connected_at = time.time()

            if self._circuit_breaker:
                self._circuit_breaker.record_success()

            await self._fire_hook(self._on_connect)
            self._start_health_check()
            await self._flush_buffer()
            return True

        except Exception as exc:  # noqa: BLE001 - must catch all to trigger reconnect
            logger.warning("[ReliableConnection] Connect failed: %s", exc)
            if self._circuit_breaker:
                self._circuit_breaker.record_failure()
            self._set_state(ConnectionState.DISCONNECTED)
            return False

    async def disconnect(self) -> None:
        """Gracefully disconnect."""
        if self._state == ConnectionState.DISCONNECTED:
            return

        self._set_state(ConnectionState.DISCONNECTING)
        self._stop_health_check()

        try:
            await self._do_disconnect()
        except Exception as exc:  # noqa: BLE001 - best-effort disconnect
            logger.debug("[ReliableConnection] Error during disconnect: %s", exc)

        self._set_state(ConnectionState.DISCONNECTED)
        await self._fire_hook(self._on_disconnect, None)

    async def reconnect(self) -> bool:
        """Attempt to reconnect using the configured policy.

        Retries with exponential backoff up to ``policy.max_retries``.
        Returns True if reconnection succeeded.
        """
        if self._state == ConnectionState.CONNECTED:
            return True

        self._set_state(ConnectionState.RECONNECTING)
        self._stop_health_check()

        for attempt in range(self._policy.max_retries):
            self._reconnect_attempts = attempt + 1

            # Check circuit breaker
            if self._circuit_breaker and not self._circuit_breaker.can_proceed():
                logger.warning(
                    "[ReliableConnection] Circuit breaker open, stopping reconnect"
                )
                self._set_state(ConnectionState.DISCONNECTED)
                return False

            delay = self._policy.calculate_delay(attempt)
            logger.info(
                "[ReliableConnection] Reconnect attempt %s/%s in %.2fs",
                attempt + 1,
                self._policy.max_retries,
                delay,
            )
            await asyncio.sleep(delay)

            try:
                await self._do_connect()
                self._set_state(ConnectionState.CONNECTED)
                self._reconnect_attempts = 0
                self._last_connected_at = time.time()

                if self._circuit_breaker:
                    self._circuit_breaker.record_success()

                await self._fire_hook(self._on_reconnect, attempt + 1)
                self._start_health_check()
                await self._flush_buffer()
                return True

            except Exception as exc:  # noqa: BLE001 - must catch all to continue retry loop
                logger.warning(
                    "[ReliableConnection] Reconnect attempt %s failed: %s",
                    attempt + 1,
                    exc,
                )
                if self._circuit_breaker:
                    self._circuit_breaker.record_failure()

        logger.error(
            "[ReliableConnection] Reconnect exhausted after %s attempts",
            self._policy.max_retries,
        )
        self._set_state(ConnectionState.DISCONNECTED)
        await self._fire_hook(self._on_disconnect, None)
        return False

    def buffer_message(self, message: Any) -> bool:
        """Buffer a message while disconnected.

        Returns True if the message was buffered, False if it was dropped
        because the buffer is full.
        """
        if len(self._buffer) >= self._buffer_size:
            self._messages_dropped += 1
            asyncio.get_event_loop().call_soon(
                lambda m=message: asyncio.ensure_future(
                    self._fire_hook(self._on_message_dropped, m)
                )
            )
            return False
        self._buffer.append(message)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Return connection statistics."""
        return {
            "state": self._state.value,
            "reconnect_attempts": self._reconnect_attempts,
            "buffered_messages": len(self._buffer),
            "messages_dropped": self._messages_dropped,
            "buffer_capacity": self._buffer_size,
            "last_connected_at": self._last_connected_at,
            "health_check_interval": self._health_check_interval,
        }

    # -- subclass interface -------------------------------------------------

    async def _do_connect(self) -> None:
        """Perform the actual connection. Override in subclasses."""
        raise NotImplementedError

    async def _do_disconnect(self) -> None:
        """Perform the actual disconnection. Override in subclasses."""
        raise NotImplementedError

    async def _do_health_check(self) -> bool:
        """Run a single health check (e.g. ping/pong).

        Return True if healthy, False otherwise. Override in subclasses.
        """
        return True

    async def _flush_buffer(self) -> int:
        """Send buffered messages after reconnection.

        Override in subclasses to actually deliver messages.
        Returns the number of flushed messages.
        """
        count = len(self._buffer)
        self._buffer.clear()
        return count

    # -- internal helpers ---------------------------------------------------

    def _set_state(self, new_state: ConnectionState) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        logger.debug(
            "[ReliableConnection] State: %s -> %s", old.value, new_state.value
        )

    def _start_health_check(self) -> None:
        if self._health_task is not None and not self._health_task.done():
            return
        self._health_task = asyncio.ensure_future(self._health_loop())

    def _stop_health_check(self) -> None:
        if self._health_task is not None and not self._health_task.done():
            self._health_task.cancel()
            self._health_task = None

    async def _health_loop(self) -> None:
        """Periodic health check loop. Triggers reconnect on failure."""
        try:
            while self._state == ConnectionState.CONNECTED:
                await asyncio.sleep(self._health_check_interval)
                if self._state != ConnectionState.CONNECTED:
                    break
                try:
                    healthy = await self._do_health_check()
                    if not healthy:
                        logger.warning(
                            "[ReliableConnection] Health check failed, reconnecting"
                        )
                        await self._fire_hook(self._on_disconnect, None)
                        await self.reconnect()
                        break
                except Exception as exc:  # noqa: BLE001 - health check must not crash loop
                    logger.warning(
                        "[ReliableConnection] Health check error: %s", exc
                    )
                    await self._fire_hook(self._on_disconnect, exc)
                    await self.reconnect()
                    break
        except asyncio.CancelledError:
            pass

    @staticmethod
    async def _fire_hook(hook: Callable[..., Any] | None, *args: Any) -> None:
        """Invoke an event hook, tolerating both sync and async callables."""
        if hook is None:
            return
        try:
            result = hook(*args)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001 - hooks must never break the connection
            logger.debug("[ReliableConnection] Hook error: %s", exc)


# ---------------------------------------------------------------------------
# WebSocket implementation
# ---------------------------------------------------------------------------


class ReliableWebSocket(ReliableConnection):
    """Reliable WebSocket client with automatic reconnection.

    Wraps the ``websockets`` library (lazy import) and provides message
    buffering, health-check ping/pong, and integration with CircuitBreaker.

    Usage:
        ws = ReliableWebSocket("ws://localhost:8765")
        await ws.connect()
        await ws.send("hello")
        msg = await ws.recv()
        await ws.disconnect()
    """

    def __init__(
        self,
        url: str,
        *,
        policy: ReconnectPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        buffer_size: int = 1000,
        health_check_interval: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy=policy,
            circuit_breaker=circuit_breaker,
            buffer_size=buffer_size,
            health_check_interval=health_check_interval,
            **kwargs,
        )
        self._url = url
        self._ws: Any | None = None
        self._extra_headers = extra_headers or {}

    @property
    def url(self) -> str:
        return self._url

    async def _do_connect(self) -> None:
        import websockets  # lazy import

        self._ws = await websockets.connect(
            self._url,
            additional_headers=self._extra_headers,
        )

    async def _do_disconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _do_health_check(self) -> bool:
        if self._ws is None:
            return False
        try:
            pong = await self._ws.ping()
            await asyncio.wait_for(pong, timeout=10.0)
            return True
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            return False

    async def send(self, message: Any) -> bool:
        """Send a message. Buffers if disconnected.

        Returns True if sent immediately, False if buffered or dropped.
        """
        if self._state != ConnectionState.CONNECTED or self._ws is None:
            return self.buffer_message(message)

        try:
            await self._ws.send(message)
            return True
        except Exception:  # noqa: BLE001 - buffer on any send failure
            self.buffer_message(message)
            # Trigger reconnect in background
            asyncio.ensure_future(self.reconnect())
            return False

    async def recv(self, timeout: float | None = None) -> Any:
        """Receive a message from the WebSocket.

        Raises ConnectionError if not connected.
        """
        if self._ws is None:
            raise ConnectionError("WebSocket is not connected")
        if timeout is not None:
            return await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        return await self._ws.recv()

    async def _flush_buffer(self) -> int:
        """Re-send buffered messages after reconnection."""
        if self._ws is None:
            return 0
        flushed = 0
        while self._buffer:
            msg = self._buffer.popleft()
            try:
                await self._ws.send(msg)
                flushed += 1
            except Exception:  # noqa: BLE001 - stop flushing on first failure
                self._buffer.appendleft(msg)
                break
        return flushed


# ---------------------------------------------------------------------------
# Kafka consumer implementation
# ---------------------------------------------------------------------------


class ReliableKafkaConsumer(ReliableConnection):
    """Reliable Kafka consumer with automatic reconnection.

    Wraps ``aiokafka.AIOKafkaConsumer`` (lazy import) and provides
    reconnection on broker failure, message buffering, and CircuitBreaker
    integration.

    Usage:
        consumer = ReliableKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topics=["events"],
        )
        await consumer.connect()
        async for msg in consumer.consume():
            process(msg)
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topics: list[str] | None = None,
        group_id: str = "aragora-reliable",
        *,
        policy: ReconnectPolicy | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        buffer_size: int = 5000,
        health_check_interval: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            policy=policy,
            circuit_breaker=circuit_breaker,
            buffer_size=buffer_size,
            health_check_interval=health_check_interval,
            **kwargs,
        )
        self._bootstrap_servers = bootstrap_servers
        self._topics = topics or ["aragora-events"]
        self._group_id = group_id
        self._consumer: Any | None = None

    async def _do_connect(self) -> None:
        from aiokafka import AIOKafkaConsumer  # lazy import

        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=self._bootstrap_servers,
            group_id=self._group_id,
        )
        await self._consumer.start()

    async def _do_disconnect(self) -> None:
        if self._consumer is not None:
            await self._consumer.stop()
            self._consumer = None

    async def _do_health_check(self) -> bool:
        """Check if the consumer is still connected to the broker."""
        if self._consumer is None:
            return False
        try:
            # Listing topics is a lightweight broker round-trip
            await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._consumer._client.cluster.topics()
                ),
                timeout=10.0,
            )
            return True
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            return False

    async def consume(self, max_messages: int | None = None) -> Any:
        """Consume messages with automatic reconnection on failure.

        Yields messages from the Kafka consumer. If the connection drops,
        triggers reconnection and resumes consuming.
        """
        if self._consumer is None:
            connected = await self.connect()
            if not connected:
                raise ConnectionError("Failed to connect to Kafka")

        consumed = 0
        while True:
            if self._consumer is None:
                success = await self.reconnect()
                if not success:
                    return

            try:
                msg = await asyncio.wait_for(
                    self._consumer.__anext__(),
                    timeout=self._health_check_interval,
                )
                yield msg
                consumed += 1
                if max_messages and consumed >= max_messages:
                    return
            except asyncio.TimeoutError:
                # No messages within interval -- that is fine, loop back
                continue
            except StopAsyncIteration:
                return
            except Exception as exc:  # noqa: BLE001 - reconnect on any consumer error
                logger.warning(
                    "[ReliableKafkaConsumer] Consume error, reconnecting: %s", exc
                )
                self._consumer = None
                self._set_state(ConnectionState.DISCONNECTED)
                success = await self.reconnect()
                if not success:
                    return


__all__ = [
    "ConnectionState",
    "ReconnectPolicy",
    "ReliableConnection",
    "ReliableKafkaConsumer",
    "ReliableWebSocket",
]
