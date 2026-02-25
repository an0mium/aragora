"""
Streaming connection reliability and reconnection hardening.

Provides automatic reconnection with exponential backoff, connection state
machine, health check ping/pong, and message buffering during reconnection
for WebSocket and enterprise streaming connections.

Key features:
- Exponential backoff with jitter (1s, 2s, 4s, 8s, 16s, max 30s by default)
- Sequence-aware reconnection: sends ``replay_from_seq`` on reconnect
- Connection quality metrics (latency, reconnect count, message loss rate)
- Heartbeat monitoring with configurable interval/timeout
- Graceful degradation hooks (e.g. fall back to HTTP polling)
- CircuitBreaker integration from aragora.resilience

Usage:
    from aragora.streaming.reliability import (
        ReconnectPolicy,
        ReliableWebSocket,
        ReliableKafkaConsumer,
        ConnectionQualityMetrics,
    )

    # WebSocket with automatic reconnection
    policy = ReconnectPolicy(max_retries=10, base_delay=1.0, max_delay=30.0)
    ws = ReliableWebSocket("ws://localhost:8765", policy=policy)
    await ws.connect()

    # Check connection quality
    quality = ws.get_quality_metrics()
    print(f"Reconnects: {quality.reconnect_count}, Avg latency: {quality.avg_latency_ms}ms")

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
import json
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

    Default backoff schedule (with jitter disabled for clarity):
        attempt 0: 1s, attempt 1: 2s, attempt 2: 4s, attempt 3: 8s,
        attempt 4: 16s, attempt 5+: 30s (capped)

    Attributes:
        max_retries: Maximum number of reconnection attempts before giving up.
            Set to 0 to disable automatic reconnection.
        base_delay: Initial delay in seconds before the first reconnection attempt.
        max_delay: Upper-bound cap on the computed backoff delay.
        backoff_factor: Multiplier applied per attempt (delay = base_delay * factor^attempt).
        jitter: Whether to add random jitter to prevent thundering herd.
    """

    max_retries: int = 10
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    jitter: bool = True

    def calculate_delay(self, attempt: int) -> float:
        """Return the backoff delay for the given attempt number (0-indexed).

        Uses exponential backoff with optional jitter:
            delay = min(base_delay * backoff_factor^attempt, max_delay)
            if jitter: delay *= uniform(0.5, 1.0)
        """
        delay = self.base_delay * (self.backoff_factor**attempt)
        delay = min(delay, self.max_delay)
        if self.jitter:
            delay *= random.uniform(0.5, 1.0)
        return max(0.0, delay)


# ---------------------------------------------------------------------------
# Connection quality metrics
# ---------------------------------------------------------------------------


@dataclass
class ConnectionQualityMetrics:
    """Snapshot of connection quality metrics.

    Provides a structured view of connection health that clients and
    monitoring systems can use to detect degradation.
    """

    reconnect_count: int = 0
    total_messages_sent: int = 0
    total_messages_buffered: int = 0
    messages_dropped: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    last_seen_seq: int = 0
    total_replayed: int = 0
    uptime_seconds: float = 0.0
    state: str = "disconnected"

    @property
    def message_loss_rate(self) -> float:
        """Fraction of messages dropped vs total attempted."""
        total = self.total_messages_sent + self.messages_dropped
        if total == 0:
            return 0.0
        return self.messages_dropped / total

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON transport."""
        return {
            "reconnect_count": self.reconnect_count,
            "total_messages_sent": self.total_messages_sent,
            "total_messages_buffered": self.total_messages_buffered,
            "messages_dropped": self.messages_dropped,
            "message_loss_rate": round(self.message_loss_rate, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "last_seen_seq": self.last_seen_seq,
            "total_replayed": self.total_replayed,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "state": self.state,
        }


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

        # Quality tracking
        self._total_reconnects = 0
        self._total_messages_sent = 0
        self._total_messages_buffered = 0
        self._total_replayed = 0
        self._last_seen_seq = 0
        self._latency_samples: deque[float] = deque(maxlen=100)
        self._created_at = time.time()

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
                logger.warning("[ReliableConnection] Circuit breaker open, stopping reconnect")
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

                self._total_reconnects += 1
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
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.create_task(self._fire_hook(self._on_message_dropped, message))
            return False
        self._buffer.append(message)
        self._total_messages_buffered += 1
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

    def record_latency(self, latency_ms: float) -> None:
        """Record a round-trip latency measurement."""
        self._latency_samples.append(latency_ms)

    def update_last_seen_seq(self, seq: int) -> None:
        """Update the last sequence number seen from the server."""
        if seq > self._last_seen_seq:
            self._last_seen_seq = seq

    def get_quality_metrics(self) -> ConnectionQualityMetrics:
        """Return a snapshot of connection quality metrics."""
        samples = list(self._latency_samples)
        avg_lat = sum(samples) / len(samples) if samples else 0.0
        max_lat = max(samples) if samples else 0.0
        min_lat = min(samples) if samples else 0.0
        uptime = time.time() - self._created_at
        return ConnectionQualityMetrics(
            reconnect_count=self._total_reconnects,
            total_messages_sent=self._total_messages_sent,
            total_messages_buffered=self._total_messages_buffered,
            messages_dropped=self._messages_dropped,
            avg_latency_ms=avg_lat,
            max_latency_ms=max_lat,
            min_latency_ms=min_lat,
            last_seen_seq=self._last_seen_seq,
            total_replayed=self._total_replayed,
            uptime_seconds=uptime,
            state=self._state.value,
        )

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
        logger.debug("[ReliableConnection] State: %s -> %s", old.value, new_state.value)

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
                        logger.warning("[ReliableConnection] Health check failed, reconnecting")
                        await self._fire_hook(self._on_disconnect, None)
                        await self.reconnect()
                        break
                except Exception as exc:  # noqa: BLE001 - health check must not crash loop
                    logger.warning("[ReliableConnection] Health check error: %s", exc)
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
    buffering, health-check ping/pong, sequence-aware reconnection with
    event replay, and integration with CircuitBreaker.

    On reconnect, sends ``replay_from_seq`` in the subscribe message
    so the server can replay missed events from its ring buffer.

    Usage:
        ws = ReliableWebSocket("ws://localhost:8765")
        await ws.connect()
        await ws.send("hello")
        msg = await ws.recv()
        quality = ws.get_quality_metrics()
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
        heartbeat_timeout: float = 90.0,
        extra_headers: dict[str, str] | None = None,
        on_degraded: Callable[[], Awaitable[None] | None] | None = None,
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
        self._heartbeat_timeout = heartbeat_timeout
        self._on_degraded = on_degraded
        self._last_heartbeat_at: float = 0.0
        self._heartbeat_monitor_task: asyncio.Task[None] | None = None

    @property
    def url(self) -> str:
        return self._url

    async def _do_connect(self) -> None:
        import websockets  # lazy import

        self._ws = await websockets.connect(
            self._url,
            additional_headers=self._extra_headers,
        )
        self._last_heartbeat_at = time.time()

    async def _do_disconnect(self) -> None:
        self._stop_heartbeat_monitor()
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
            self.buffer_message(message)
            return False

        try:
            await self._ws.send(message)
            self._total_messages_sent += 1
            return True
        except Exception:  # noqa: BLE001 - buffer on any send failure
            self.buffer_message(message)
            # Trigger reconnect in background
            asyncio.ensure_future(self.reconnect())
            return False

    async def recv(self, timeout: float | None = None) -> Any:
        """Receive a message from the WebSocket.

        Raises ConnectionError if not connected.
        Automatically updates last_seen_seq and heartbeat timestamp
        if the received message is JSON with a ``seq`` field.
        """
        if self._ws is None:
            raise ConnectionError("WebSocket is not connected")
        if timeout is not None:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
        else:
            raw = await self._ws.recv()

        # Auto-track sequence number and heartbeat from incoming events
        self._last_heartbeat_at = time.time()
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    seq = parsed.get("seq")
                    if isinstance(seq, int) and seq > 0:
                        self.update_last_seen_seq(seq)
                    if parsed.get("type") == "pong":
                        client_ts = parsed.get("data", {}).get("client_ts", 0)
                        if client_ts > 0:
                            rtt = time.time() * 1000 - client_ts
                            if rtt > 0:
                                self.record_latency(rtt)
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        return raw

    async def send_subscribe(self, debate_id: str) -> None:
        """Send a subscribe message, including replay_from_seq if reconnecting.

        This sends the appropriate subscribe payload to the Aragora debate
        stream server, requesting event replay from the last seen sequence
        number on reconnection.
        """
        msg: dict[str, Any] = {"type": "subscribe", "debate_id": debate_id}
        if self._last_seen_seq > 0:
            msg["replay_from_seq"] = self._last_seen_seq
            logger.info(
                "[ReliableWS] Requesting replay from seq %d for debate %s",
                self._last_seen_seq,
                debate_id,
            )
        await self.send(json.dumps(msg))

    async def send_ping(self) -> None:
        """Send an application-level ping for latency measurement."""
        msg = json.dumps({"type": "ping", "ts": time.time() * 1000})
        await self.send(msg)

    def _start_heartbeat_monitor(self) -> None:
        """Start monitoring for heartbeat timeout."""
        self._stop_heartbeat_monitor()
        self._heartbeat_monitor_task = asyncio.ensure_future(self._heartbeat_monitor_loop())

    def _stop_heartbeat_monitor(self) -> None:
        """Stop the heartbeat monitor task."""
        if self._heartbeat_monitor_task and not self._heartbeat_monitor_task.done():
            self._heartbeat_monitor_task.cancel()
            self._heartbeat_monitor_task = None

    async def _heartbeat_monitor_loop(self) -> None:
        """Monitor for heartbeat timeout and trigger reconnect/degradation."""
        try:
            while self._state == ConnectionState.CONNECTED:
                await asyncio.sleep(self._heartbeat_timeout / 3)
                if self._state != ConnectionState.CONNECTED:
                    break
                elapsed = time.time() - self._last_heartbeat_at
                if elapsed > self._heartbeat_timeout:
                    logger.warning(
                        "[ReliableWS] No heartbeat in %.0fs (timeout=%.0fs), triggering reconnect",
                        elapsed,
                        self._heartbeat_timeout,
                    )
                    # Try reconnect first
                    reconnected = await self.reconnect()
                    if not reconnected:
                        # Fire degradation hook (e.g. switch to polling)
                        await self._fire_hook(self._on_degraded)
                    break
        except asyncio.CancelledError:
            pass

    async def _flush_buffer(self) -> int:
        """Re-send buffered messages after reconnection."""
        if self._ws is None:
            return 0
        flushed = 0
        while self._buffer:
            msg = self._buffer.popleft()
            try:
                await self._ws.send(msg)
                self._total_messages_sent += 1
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
                asyncio.get_running_loop().run_in_executor(
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
                logger.warning("[ReliableKafkaConsumer] Consume error, reconnecting: %s", exc)
                self._consumer = None
                self._set_state(ConnectionState.DISCONNECTED)
                success = await self.reconnect()
                if not success:
                    return


__all__ = [
    "ConnectionQualityMetrics",
    "ConnectionState",
    "ReconnectPolicy",
    "ReliableConnection",
    "ReliableKafkaConsumer",
    "ReliableWebSocket",
]
