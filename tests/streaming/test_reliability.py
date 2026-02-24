"""
Tests for streaming connection reliability and reconnection hardening.

Covers:
- ReconnectPolicy backoff calculations (1s, 2s, 4s, 8s, 16s, max 30s)
- ConnectionState transitions
- Message buffering during disconnect
- Circuit breaker integration
- ReliableWebSocket send/buffer/reconnect
- ReliableKafkaConsumer connect/reconnect
- Health check triggering reconnection
- ConnectionQualityMetrics tracking
- Sequence-aware reconnection and replay
- EventReplayBuffer (server-side)
- ConnectionQualityTracker (server-side)
- Heartbeat timeout and graceful degradation
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.streaming.reliability import (
    ConnectionQualityMetrics,
    ConnectionState,
    ReconnectPolicy,
    ReliableConnection,
    ReliableKafkaConsumer,
    ReliableWebSocket,
)


# ---------------------------------------------------------------------------
# ReconnectPolicy tests
# ---------------------------------------------------------------------------


class TestReconnectPolicy:
    """Tests for ReconnectPolicy configuration and delay calculation."""

    def test_default_policy(self):
        policy = ReconnectPolicy()
        assert policy.max_retries == 10
        assert policy.base_delay == 1.0
        assert policy.max_delay == 30.0
        assert policy.backoff_factor == 2.0
        assert policy.jitter is True

    def test_custom_policy(self):
        policy = ReconnectPolicy(
            max_retries=10,
            base_delay=0.5,
            max_delay=120.0,
            backoff_factor=3.0,
            jitter=False,
        )
        assert policy.max_retries == 10
        assert policy.base_delay == 0.5

    def test_exponential_backoff_no_jitter(self):
        policy = ReconnectPolicy(
            base_delay=1.0, backoff_factor=2.0, max_delay=100.0, jitter=False
        )
        assert policy.calculate_delay(0) == 1.0  # 1 * 2^0
        assert policy.calculate_delay(1) == 2.0  # 1 * 2^1
        assert policy.calculate_delay(2) == 4.0  # 1 * 2^2
        assert policy.calculate_delay(3) == 8.0  # 1 * 2^3
        assert policy.calculate_delay(4) == 16.0  # 1 * 2^4

    def test_backoff_schedule_matches_spec(self):
        """Verify the backoff schedule: 1s, 2s, 4s, 8s, 16s, max 30s."""
        policy = ReconnectPolicy(
            base_delay=1.0, backoff_factor=2.0, max_delay=30.0, jitter=False
        )
        assert policy.calculate_delay(0) == 1.0
        assert policy.calculate_delay(1) == 2.0
        assert policy.calculate_delay(2) == 4.0
        assert policy.calculate_delay(3) == 8.0
        assert policy.calculate_delay(4) == 16.0
        assert policy.calculate_delay(5) == 30.0  # capped
        assert policy.calculate_delay(6) == 30.0  # still capped

    def test_backoff_respects_max_delay(self):
        policy = ReconnectPolicy(
            base_delay=1.0, backoff_factor=2.0, max_delay=5.0, jitter=False
        )
        assert policy.calculate_delay(0) == 1.0
        assert policy.calculate_delay(10) == 5.0  # capped at max_delay

    def test_jitter_reduces_delay(self):
        """Jitter multiplies by uniform(0.5, 1.0), so delay <= no-jitter delay."""
        policy = ReconnectPolicy(
            base_delay=10.0, backoff_factor=1.0, max_delay=100.0, jitter=True
        )
        for _ in range(50):
            delay = policy.calculate_delay(0)
            assert 0.0 <= delay <= 10.0

    def test_delay_never_negative(self):
        policy = ReconnectPolicy(base_delay=0.01, max_delay=0.01, jitter=True)
        for attempt in range(10):
            assert policy.calculate_delay(attempt) >= 0.0


# ---------------------------------------------------------------------------
# ConnectionState tests
# ---------------------------------------------------------------------------


class TestConnectionState:
    """Tests for the ConnectionState enum."""

    def test_all_states_exist(self):
        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"
        assert ConnectionState.DISCONNECTING == "disconnecting"


# ---------------------------------------------------------------------------
# ConnectionQualityMetrics tests
# ---------------------------------------------------------------------------


class TestConnectionQualityMetrics:
    """Tests for ConnectionQualityMetrics dataclass."""

    def test_default_metrics(self):
        m = ConnectionQualityMetrics()
        assert m.reconnect_count == 0
        assert m.total_messages_sent == 0
        assert m.messages_dropped == 0
        assert m.message_loss_rate == 0.0
        assert m.state == "disconnected"

    def test_message_loss_rate(self):
        m = ConnectionQualityMetrics(total_messages_sent=90, messages_dropped=10)
        assert abs(m.message_loss_rate - 0.1) < 0.001

    def test_zero_messages_loss_rate(self):
        m = ConnectionQualityMetrics(total_messages_sent=0, messages_dropped=0)
        assert m.message_loss_rate == 0.0

    def test_to_dict(self):
        m = ConnectionQualityMetrics(
            reconnect_count=3,
            total_messages_sent=100,
            messages_dropped=5,
            avg_latency_ms=42.5,
            state="connected",
        )
        d = m.to_dict()
        assert d["reconnect_count"] == 3
        assert d["total_messages_sent"] == 100
        assert d["messages_dropped"] == 5
        assert abs(d["message_loss_rate"] - 5 / 105) < 0.001
        assert d["avg_latency_ms"] == 42.5
        assert d["state"] == "connected"

    def test_to_dict_all_fields_present(self):
        m = ConnectionQualityMetrics()
        d = m.to_dict()
        expected_keys = {
            "reconnect_count",
            "total_messages_sent",
            "total_messages_buffered",
            "messages_dropped",
            "message_loss_rate",
            "avg_latency_ms",
            "max_latency_ms",
            "min_latency_ms",
            "last_seen_seq",
            "total_replayed",
            "uptime_seconds",
            "state",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# ReliableConnection base tests
# ---------------------------------------------------------------------------


class ConcreteConnection(ReliableConnection):
    """Concrete subclass for testing the base class."""

    def __init__(self, fail_connect: bool = False, fail_health: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.fail_connect = fail_connect
        self.fail_health = fail_health
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.health_checks = 0
        self.flushed_messages: list = []

    async def _do_connect(self):
        self.connect_calls += 1
        if self.fail_connect:
            raise ConnectionError("connection refused")

    async def _do_disconnect(self):
        self.disconnect_calls += 1

    async def _do_health_check(self) -> bool:
        self.health_checks += 1
        return not self.fail_health

    async def _flush_buffer(self) -> int:
        count = len(self._buffer)
        self.flushed_messages.extend(self._buffer)
        self._buffer.clear()
        return count


class TestReliableConnection:
    """Tests for ReliableConnection state machine and reconnection."""

    @pytest.mark.asyncio
    async def test_initial_state_is_disconnected(self):
        conn = ConcreteConnection()
        assert conn.state == ConnectionState.DISCONNECTED
        assert not conn.is_connected

    @pytest.mark.asyncio
    async def test_connect_transitions_to_connected(self):
        conn = ConcreteConnection()
        result = await conn.connect()
        assert result is True
        assert conn.state == ConnectionState.CONNECTED
        assert conn.is_connected
        assert conn.connect_calls == 1
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_connect_failure_stays_disconnected(self):
        conn = ConcreteConnection(fail_connect=True)
        result = await conn.connect()
        assert result is False
        assert conn.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_transitions_to_disconnected(self):
        conn = ConcreteConnection()
        await conn.connect()
        assert conn.state == ConnectionState.CONNECTED
        await conn.disconnect()
        assert conn.state == ConnectionState.DISCONNECTED
        assert conn.disconnect_calls == 1

    @pytest.mark.asyncio
    async def test_disconnect_when_already_disconnected_is_noop(self):
        conn = ConcreteConnection()
        await conn.disconnect()
        assert conn.disconnect_calls == 0

    @pytest.mark.asyncio
    async def test_reconnect_after_failure(self):
        conn = ConcreteConnection(
            fail_connect=False,
            policy=ReconnectPolicy(max_retries=3, base_delay=0.01, jitter=False),
        )
        # Start disconnected, reconnect should work
        result = await conn.reconnect()
        assert result is True
        assert conn.state == ConnectionState.CONNECTED
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_reconnect_exhausts_retries(self):
        conn = ConcreteConnection(
            fail_connect=True,
            policy=ReconnectPolicy(max_retries=3, base_delay=0.01, jitter=False),
        )
        result = await conn.reconnect()
        assert result is False
        assert conn.state == ConnectionState.DISCONNECTED
        assert conn.connect_calls == 3  # tried 3 times

    @pytest.mark.asyncio
    async def test_reconnect_succeeds_on_second_attempt(self):
        conn = ConcreteConnection(
            fail_connect=True,
            policy=ReconnectPolicy(max_retries=5, base_delay=0.01, jitter=False),
        )
        # Fail first two attempts, succeed on third
        original_do_connect = conn._do_connect

        call_count = 0

        async def flaky_connect():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ConnectionError("temporary failure")
            # Success from third attempt on
            conn.connect_calls += 1

        conn._do_connect = flaky_connect
        result = await conn.reconnect()
        assert result is True
        assert conn.state == ConnectionState.CONNECTED
        assert call_count == 3
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_connect_when_already_connected(self):
        conn = ConcreteConnection()
        await conn.connect()
        # Second connect should be a no-op that returns True
        result = await conn.connect()
        assert result is True
        assert conn.connect_calls == 1  # only the first actual connect
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_reconnect_increments_total_reconnects(self):
        conn = ConcreteConnection(
            policy=ReconnectPolicy(max_retries=3, base_delay=0.01, jitter=False),
        )
        await conn.reconnect()
        assert conn._total_reconnects == 1
        await conn.disconnect()

        # Reconnect again
        await conn.reconnect()
        assert conn._total_reconnects == 2
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_quality_metrics_after_lifecycle(self):
        conn = ConcreteConnection(
            buffer_size=100,
            policy=ReconnectPolicy(max_retries=3, base_delay=0.01, jitter=False),
        )
        conn.buffer_message("msg1")
        conn.buffer_message("msg2")
        await conn.reconnect()

        conn.record_latency(10.0)
        conn.record_latency(20.0)
        conn.update_last_seen_seq(42)

        metrics = conn.get_quality_metrics()
        assert metrics.reconnect_count == 1
        assert metrics.total_messages_buffered == 2
        assert metrics.avg_latency_ms == 15.0
        assert metrics.max_latency_ms == 20.0
        assert metrics.min_latency_ms == 10.0
        assert metrics.last_seen_seq == 42
        assert metrics.state == "connected"
        assert metrics.uptime_seconds > 0
        await conn.disconnect()


# ---------------------------------------------------------------------------
# Message buffering tests
# ---------------------------------------------------------------------------


class TestMessageBuffering:
    """Tests for message buffering during disconnection."""

    @pytest.mark.asyncio
    async def test_buffer_message_when_disconnected(self):
        conn = ConcreteConnection(buffer_size=10)
        assert conn.buffer_message("msg1") is True
        assert conn.buffer_message("msg2") is True
        assert conn.buffered_count == 2

    @pytest.mark.asyncio
    async def test_buffer_overflow_drops_messages(self):
        conn = ConcreteConnection(buffer_size=2)
        assert conn.buffer_message("msg1") is True
        assert conn.buffer_message("msg2") is True
        assert conn.buffer_message("msg3") is False  # dropped
        assert conn.buffered_count == 2
        assert conn.messages_dropped == 1

    @pytest.mark.asyncio
    async def test_flush_on_reconnect(self):
        conn = ConcreteConnection(
            buffer_size=100,
            policy=ReconnectPolicy(max_retries=3, base_delay=0.01, jitter=False),
        )
        conn.buffer_message("msg1")
        conn.buffer_message("msg2")
        assert conn.buffered_count == 2

        await conn.reconnect()
        assert conn.state == ConnectionState.CONNECTED
        assert conn.buffered_count == 0
        assert conn.flushed_messages == ["msg1", "msg2"]
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_on_message_dropped_hook(self):
        dropped = []

        async def on_drop(msg):
            dropped.append(msg)

        conn = ConcreteConnection(buffer_size=1, on_message_dropped=on_drop)
        conn.buffer_message("keep")
        conn.buffer_message("drop_me")  # triggers hook
        assert conn.messages_dropped == 1
        # Give event loop a chance to fire the hook
        await asyncio.sleep(0.05)
        assert "drop_me" in dropped

    @pytest.mark.asyncio
    async def test_get_stats(self):
        conn = ConcreteConnection(buffer_size=50)
        conn.buffer_message("msg1")
        stats = conn.get_stats()
        assert stats["state"] == "disconnected"
        assert stats["buffered_messages"] == 1
        assert stats["messages_dropped"] == 0
        assert stats["buffer_capacity"] == 50

    @pytest.mark.asyncio
    async def test_buffer_tracks_total_buffered(self):
        conn = ConcreteConnection(buffer_size=100)
        conn.buffer_message("a")
        conn.buffer_message("b")
        conn.buffer_message("c")
        metrics = conn.get_quality_metrics()
        assert metrics.total_messages_buffered == 3


# ---------------------------------------------------------------------------
# Circuit breaker integration tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    """Tests for CircuitBreaker integration with ReliableConnection."""

    @pytest.mark.asyncio
    async def test_connect_blocked_by_open_circuit(self):
        from aragora.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test-stream", failure_threshold=1, cooldown_seconds=300)
        cb.record_failure()  # open the circuit

        conn = ConcreteConnection(circuit_breaker=cb)
        result = await conn.connect()
        assert result is False
        assert conn.state == ConnectionState.DISCONNECTED
        assert conn.connect_calls == 0  # never tried

    @pytest.mark.asyncio
    async def test_success_records_to_circuit_breaker(self):
        from aragora.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test-stream", failure_threshold=3)
        conn = ConcreteConnection(circuit_breaker=cb)

        await conn.connect()
        assert cb.failures == 0
        assert cb.state == "closed"
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_failure_records_to_circuit_breaker(self):
        from aragora.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test-stream", failure_threshold=3)
        conn = ConcreteConnection(fail_connect=True, circuit_breaker=cb)

        await conn.connect()
        assert cb.failures == 1

    @pytest.mark.asyncio
    async def test_reconnect_stops_when_circuit_opens(self):
        from aragora.resilience.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test-stream", failure_threshold=2, cooldown_seconds=300)
        conn = ConcreteConnection(
            fail_connect=True,
            circuit_breaker=cb,
            policy=ReconnectPolicy(max_retries=10, base_delay=0.01, jitter=False),
        )

        result = await conn.reconnect()
        assert result is False
        # Should have stopped after circuit opened (after 2 failures)
        # The exact count depends on when the circuit check happens
        assert conn.connect_calls <= 3


# ---------------------------------------------------------------------------
# Event hooks tests
# ---------------------------------------------------------------------------


class TestEventHooks:
    """Tests for on_connect / on_disconnect / on_reconnect hooks."""

    @pytest.mark.asyncio
    async def test_on_connect_hook_fires(self):
        called = []

        async def on_connect():
            called.append("connected")

        conn = ConcreteConnection(on_connect=on_connect)
        await conn.connect()
        assert "connected" in called
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_on_disconnect_hook_fires(self):
        called = []

        async def on_disconnect(exc):
            called.append(("disconnected", exc))

        conn = ConcreteConnection(on_disconnect=on_disconnect)
        await conn.connect()
        await conn.disconnect()
        assert len(called) == 1
        assert called[0][0] == "disconnected"

    @pytest.mark.asyncio
    async def test_on_reconnect_hook_fires(self):
        called = []

        async def on_reconnect(attempt):
            called.append(attempt)

        conn = ConcreteConnection(
            on_reconnect=on_reconnect,
            policy=ReconnectPolicy(max_retries=3, base_delay=0.01, jitter=False),
        )
        await conn.reconnect()
        assert len(called) == 1
        assert called[0] == 1  # succeeded on first attempt

    @pytest.mark.asyncio
    async def test_sync_hook_works(self):
        """Hooks can be sync functions too."""
        called = []

        def on_connect():
            called.append("sync_connected")

        conn = ConcreteConnection(on_connect=on_connect)
        await conn.connect()
        assert "sync_connected" in called
        await conn.disconnect()


# ---------------------------------------------------------------------------
# ReliableWebSocket tests
# ---------------------------------------------------------------------------


class TestReliableWebSocket:
    """Tests for ReliableWebSocket."""

    @pytest.mark.asyncio
    async def test_initial_state(self):
        ws = ReliableWebSocket("ws://localhost:8765")
        assert ws.url == "ws://localhost:8765"
        assert ws.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_send_buffers_when_disconnected(self):
        ws = ReliableWebSocket("ws://localhost:8765", buffer_size=10)
        result = await ws.send("hello")
        assert result is False  # buffered, not sent
        assert ws.buffered_count == 1

    @pytest.mark.asyncio
    async def test_recv_raises_when_disconnected(self):
        ws = ReliableWebSocket("ws://localhost:8765")
        with pytest.raises(ConnectionError, match="not connected"):
            await ws.recv()

    @pytest.mark.asyncio
    async def test_connect_and_send(self):
        """Test connect + send with mocked websockets library."""
        mock_ws = AsyncMock()
        mock_ws.ping = AsyncMock(return_value=asyncio.Future())
        mock_ws.ping.return_value.set_result(None)

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect") as mock_connect:
            mock_connect.return_value = None
            ws = ReliableWebSocket("ws://localhost:8765")
            ws._ws = mock_ws

            await ws.connect()
            assert ws.state == ConnectionState.CONNECTED

            result = await ws.send("test message")
            mock_ws.send.assert_called_once_with("test message")
            assert result is True

            await ws.disconnect()

    @pytest.mark.asyncio
    async def test_send_tracks_messages_sent(self):
        """Verify that successful sends increment total_messages_sent."""
        mock_ws = AsyncMock()

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect"):
            ws = ReliableWebSocket("ws://localhost:8765")
            ws._ws = mock_ws
            await ws.connect()

            await ws.send("msg1")
            await ws.send("msg2")
            await ws.send("msg3")

            metrics = ws.get_quality_metrics()
            assert metrics.total_messages_sent == 3
            assert metrics.state == "connected"
            await ws.disconnect()

    @pytest.mark.asyncio
    async def test_recv_updates_seq_from_json(self):
        """Verify recv() auto-tracks seq from JSON messages."""
        mock_ws = AsyncMock()
        event = json.dumps({"type": "agent_message", "seq": 42, "data": {}})
        mock_ws.recv = AsyncMock(return_value=event)

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect"):
            ws = ReliableWebSocket("ws://localhost:8765")
            ws._ws = mock_ws
            await ws.connect()

            result = await ws.recv()
            assert result == event
            assert ws._last_seen_seq == 42
            await ws.disconnect()

    @pytest.mark.asyncio
    async def test_recv_tracks_pong_latency(self):
        """Verify recv() records latency from pong messages."""
        now_ms = time.time() * 1000
        pong_event = json.dumps({
            "type": "pong",
            "data": {"client_ts": now_ms - 50, "server_ts": now_ms - 25},
        })
        mock_ws = AsyncMock()
        mock_ws.recv = AsyncMock(return_value=pong_event)

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect"):
            ws = ReliableWebSocket("ws://localhost:8765")
            ws._ws = mock_ws
            await ws.connect()

            await ws.recv()
            metrics = ws.get_quality_metrics()
            # Should have at least one latency sample
            assert len(ws._latency_samples) == 1
            assert metrics.avg_latency_ms > 0
            await ws.disconnect()

    @pytest.mark.asyncio
    async def test_send_subscribe_fresh(self):
        """send_subscribe sends plain subscribe without replay_from_seq on fresh connect."""
        mock_ws = AsyncMock()

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect"):
            ws = ReliableWebSocket("ws://localhost:8765")
            ws._ws = mock_ws
            await ws.connect()

            await ws.send_subscribe("debate-123")
            # Check what was sent
            sent_json = mock_ws.send.call_args[0][0]
            sent = json.loads(sent_json)
            assert sent["type"] == "subscribe"
            assert sent["debate_id"] == "debate-123"
            assert "replay_from_seq" not in sent
            await ws.disconnect()

    @pytest.mark.asyncio
    async def test_send_subscribe_with_replay(self):
        """send_subscribe includes replay_from_seq when reconnecting."""
        mock_ws = AsyncMock()

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect"):
            ws = ReliableWebSocket("ws://localhost:8765")
            ws._ws = mock_ws
            await ws.connect()

            # Simulate having received events
            ws.update_last_seen_seq(99)

            await ws.send_subscribe("debate-123")
            sent_json = mock_ws.send.call_args[0][0]
            sent = json.loads(sent_json)
            assert sent["type"] == "subscribe"
            assert sent["debate_id"] == "debate-123"
            assert sent["replay_from_seq"] == 99
            await ws.disconnect()

    @pytest.mark.asyncio
    async def test_send_ping(self):
        """send_ping sends a JSON ping with timestamp."""
        mock_ws = AsyncMock()

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect"):
            ws = ReliableWebSocket("ws://localhost:8765")
            ws._ws = mock_ws
            await ws.connect()

            await ws.send_ping()
            sent_json = mock_ws.send.call_args[0][0]
            sent = json.loads(sent_json)
            assert sent["type"] == "ping"
            assert "ts" in sent
            assert sent["ts"] > 0
            await ws.disconnect()

    @pytest.mark.asyncio
    async def test_quality_metrics_comprehensive(self):
        """Verify quality metrics reflect full lifecycle."""
        mock_ws = AsyncMock()

        with patch("aragora.streaming.reliability.ReliableWebSocket._do_connect"):
            ws = ReliableWebSocket("ws://localhost:8765", buffer_size=5)
            ws._ws = mock_ws
            await ws.connect()

            # Send messages
            await ws.send("m1")
            await ws.send("m2")

            # Record latency
            ws.record_latency(10.0)
            ws.record_latency(30.0)

            # Update seq
            ws.update_last_seen_seq(50)

            metrics = ws.get_quality_metrics()
            assert metrics.total_messages_sent == 2
            assert metrics.avg_latency_ms == 20.0
            assert metrics.max_latency_ms == 30.0
            assert metrics.min_latency_ms == 10.0
            assert metrics.last_seen_seq == 50
            assert metrics.state == "connected"
            assert metrics.reconnect_count == 0
            assert metrics.messages_dropped == 0
            assert metrics.message_loss_rate == 0.0
            await ws.disconnect()


# ---------------------------------------------------------------------------
# ReliableKafkaConsumer tests
# ---------------------------------------------------------------------------


class TestReliableKafkaConsumer:
    """Tests for ReliableKafkaConsumer."""

    @pytest.mark.asyncio
    async def test_initial_state(self):
        consumer = ReliableKafkaConsumer(
            bootstrap_servers="localhost:9092", topics=["test"]
        )
        assert consumer.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_with_mock(self):
        consumer = ReliableKafkaConsumer(
            bootstrap_servers="localhost:9092", topics=["test"]
        )

        with patch(
            "aragora.streaming.reliability.ReliableKafkaConsumer._do_connect"
        ) as mock_connect:
            mock_connect.return_value = None
            result = await consumer.connect()
            assert result is True
            assert consumer.state == ConnectionState.CONNECTED
            await consumer.disconnect()

    @pytest.mark.asyncio
    async def test_reconnect_on_consume_error(self):
        """Consume should reconnect if the consumer raises an error."""
        consumer = ReliableKafkaConsumer(
            bootstrap_servers="localhost:9092",
            topics=["test"],
            policy=ReconnectPolicy(max_retries=2, base_delay=0.01, jitter=False),
        )

        # Set up a mock consumer that fails then stops
        mock_kafka_consumer = AsyncMock()
        call_count = 0

        async def mock_anext():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("broker gone")
            raise StopAsyncIteration

        mock_kafka_consumer.__anext__ = mock_anext
        # Mock _do_connect to succeed and restore the consumer after reconnect.
        connect_count = 0

        async def mock_connect():
            nonlocal connect_count
            connect_count += 1
            consumer._consumer = mock_kafka_consumer

        consumer._do_connect = mock_connect
        consumer._consumer = mock_kafka_consumer

        # Mark as connected
        consumer._set_state(ConnectionState.CONNECTED)

        messages = []
        async for msg in consumer.consume():
            messages.append(msg)

        # Should have tried to reconnect
        assert connect_count >= 1


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


class TestStateTransitions:
    """Tests for correct state transitions through the lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        states: list[ConnectionState] = []

        conn = ConcreteConnection(
            policy=ReconnectPolicy(max_retries=2, base_delay=0.01, jitter=False),
        )

        # Track state changes
        original_set = conn._set_state

        def tracking_set(new_state):
            states.append(new_state)
            original_set(new_state)

        conn._set_state = tracking_set

        # Connect
        await conn.connect()
        assert ConnectionState.CONNECTING in states
        assert ConnectionState.CONNECTED in states

        # Disconnect
        await conn.disconnect()
        assert ConnectionState.DISCONNECTING in states
        assert ConnectionState.DISCONNECTED in states

    @pytest.mark.asyncio
    async def test_reconnect_goes_through_reconnecting_state(self):
        states: list[ConnectionState] = []

        conn = ConcreteConnection(
            policy=ReconnectPolicy(max_retries=2, base_delay=0.01, jitter=False),
        )
        original_set = conn._set_state

        def tracking_set(new_state):
            states.append(new_state)
            original_set(new_state)

        conn._set_state = tracking_set

        await conn.reconnect()
        assert ConnectionState.RECONNECTING in states
        assert ConnectionState.CONNECTED in states
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_failed_reconnect_ends_disconnected(self):
        conn = ConcreteConnection(
            fail_connect=True,
            policy=ReconnectPolicy(max_retries=2, base_delay=0.01, jitter=False),
        )
        await conn.reconnect()
        assert conn.state == ConnectionState.DISCONNECTED


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """Tests for health check loop behavior."""

    @pytest.mark.asyncio
    async def test_health_check_starts_on_connect(self):
        conn = ConcreteConnection(health_check_interval=0.05)
        await conn.connect()
        assert conn._health_task is not None
        assert not conn._health_task.done()
        await conn.disconnect()

    @pytest.mark.asyncio
    async def test_health_check_stops_on_disconnect(self):
        conn = ConcreteConnection(health_check_interval=0.05)
        await conn.connect()
        task = conn._health_task
        await conn.disconnect()
        # Give cancelled task a moment to finish
        await asyncio.sleep(0.02)
        assert task is not None and task.cancelled()

    @pytest.mark.asyncio
    async def test_health_failure_triggers_reconnect(self):
        conn = ConcreteConnection(
            health_check_interval=0.05,
            policy=ReconnectPolicy(max_retries=2, base_delay=0.01, jitter=False),
        )
        await conn.connect()
        assert conn.state == ConnectionState.CONNECTED

        # Now make health check fail
        conn.fail_health = True

        # Wait for health check to fire and trigger reconnect
        await asyncio.sleep(0.2)

        # After reconnect, should be connected again (connect succeeds)
        # The health loop should have detected failure and reconnected
        assert conn.health_checks >= 1
        await conn.disconnect()


# ---------------------------------------------------------------------------
# EventReplayBuffer tests (server-side)
# ---------------------------------------------------------------------------


class TestEventReplayBuffer:
    """Tests for the server-side event replay buffer."""

    def _make_event(self, loop_id: str = "debate-1", seq: int = 1, event_type: str = "agent_message"):
        """Create a minimal StreamEvent for testing."""
        from aragora.events.types import StreamEvent, StreamEventType

        return StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"content": f"message-{seq}"},
            loop_id=loop_id,
            seq=seq,
            agent=f"agent-{seq % 3}",
        )

    def test_append_and_replay(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer(max_per_debate=100)

        for i in range(1, 6):
            buf.append(self._make_event(seq=i))

        # Replay from seq 2 should return events 3, 4, 5
        replayed = buf.replay_since("debate-1", 2)
        assert len(replayed) == 3

        # Verify they are valid JSON
        for r in replayed:
            parsed = json.loads(r)
            assert "seq" in parsed
            assert parsed["seq"] > 2

    def test_replay_empty_buffer(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer()
        assert buf.replay_since("nonexistent", 0) == []

    def test_get_latest_seq(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer()
        assert buf.get_latest_seq("debate-1") == 0

        for i in range(1, 4):
            buf.append(self._make_event(seq=i))

        assert buf.get_latest_seq("debate-1") == 3

    def test_get_oldest_seq(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer(max_per_debate=3)
        assert buf.get_oldest_seq("debate-1") == 0

        for i in range(1, 6):
            buf.append(self._make_event(seq=i))

        # Only last 3 events should be retained (3, 4, 5)
        assert buf.get_oldest_seq("debate-1") == 3

    def test_get_buffered_count(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer(max_per_debate=100)
        assert buf.get_buffered_count("debate-1") == 0

        for i in range(1, 4):
            buf.append(self._make_event(seq=i))

        assert buf.get_buffered_count("debate-1") == 3

    def test_buffer_bounded_to_max(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer(max_per_debate=5)

        for i in range(1, 11):
            buf.append(self._make_event(seq=i))

        # Should only keep last 5
        assert buf.get_buffered_count("debate-1") == 5
        assert buf.get_oldest_seq("debate-1") == 6
        assert buf.get_latest_seq("debate-1") == 10

    def test_default_buffer_size_is_1000(self):
        from aragora.server.stream.replay_buffer import EVENT_REPLAY_BUFFER_SIZE

        assert EVENT_REPLAY_BUFFER_SIZE == 1000

    def test_separate_buffers_per_debate(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer()

        buf.append(self._make_event(loop_id="debate-1", seq=1))
        buf.append(self._make_event(loop_id="debate-1", seq=2))
        buf.append(self._make_event(loop_id="debate-2", seq=10))

        assert buf.get_buffered_count("debate-1") == 2
        assert buf.get_buffered_count("debate-2") == 1
        assert buf.get_latest_seq("debate-1") == 2
        assert buf.get_latest_seq("debate-2") == 10

    def test_remove_cleans_up(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer()
        buf.append(self._make_event(seq=1))
        buf.remove("debate-1")
        assert buf.get_buffered_count("debate-1") == 0
        assert buf.replay_since("debate-1", 0) == []

    def test_cleanup_stale(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer()
        buf.append(self._make_event(loop_id="active", seq=1))
        buf.append(self._make_event(loop_id="stale", seq=1))

        removed = buf.cleanup_stale({"active"})
        assert removed == 1
        assert buf.get_buffered_count("active") == 1
        assert buf.get_buffered_count("stale") == 0

    def test_metrics_tracking(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer(max_per_debate=3)

        for i in range(1, 6):
            buf.append(self._make_event(seq=i))

        metrics = buf.get_metrics("debate-1")
        assert metrics["appended"] == 5
        assert metrics["evicted"] == 2  # 2 events evicted (4th and 5th push out 1st and 2nd)

    def test_events_without_loop_id_ignored(self):
        from aragora.events.types import StreamEvent, StreamEventType
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer()
        event = StreamEvent(
            type=StreamEventType.HEARTBEAT,
            data={},
            loop_id="",  # empty loop_id
            seq=1,
        )
        buf.append(event)
        assert buf.get_buffered_count("") == 0

    def test_replay_since_returns_ordered(self):
        from aragora.server.stream.replay_buffer import EventReplayBuffer

        buf = EventReplayBuffer()
        for i in [5, 3, 7, 1, 9]:
            buf.append(self._make_event(seq=i))

        replayed = buf.replay_since("debate-1", 3)
        seqs = [json.loads(r)["seq"] for r in replayed]
        # Should return events in insertion order that are > 3
        assert seqs == [5, 7, 9]


# ---------------------------------------------------------------------------
# ConnectionQualityTracker tests (server-side)
# ---------------------------------------------------------------------------


class TestConnectionQualityTracker:
    """Tests for the server-side per-client quality tracker."""

    def test_register_and_get(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(42)
        quality = tracker.get_quality(42)
        assert quality is not None
        assert quality["reconnect_count"] == 0
        assert quality["messages_received"] == 0

    def test_unregister(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(42)
        final = tracker.unregister(42)
        assert final is not None
        assert tracker.get_quality(42) is None

    def test_record_reconnect(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(42)
        tracker.record_reconnect(42)
        tracker.record_reconnect(42)
        quality = tracker.get_quality(42)
        assert quality["reconnect_count"] == 2
        assert quality["last_reconnect_at"] is not None

    def test_record_replay(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(42)
        tracker.record_replay(42, 15)
        tracker.record_replay(42, 5)
        quality = tracker.get_quality(42)
        assert quality["replay_requests"] == 2
        assert quality["total_replayed"] == 20

    def test_record_message_sent(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(42)
        for _ in range(10):
            tracker.record_message_sent(42)
        quality = tracker.get_quality(42)
        assert quality["messages_received"] == 10

    def test_update_last_seen_seq(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(42)
        tracker.update_last_seen_seq(42, 100)
        tracker.update_last_seen_seq(42, 50)  # should NOT decrease
        quality = tracker.get_quality(42)
        assert quality["last_seen_seq"] == 100

    def test_record_latency(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(42)
        tracker.record_latency(42, 10.0)
        tracker.record_latency(42, 30.0)
        quality = tracker.get_quality(42)
        assert quality["avg_latency_ms"] == 20.0
        assert quality["latency_sample_count"] == 2

    def test_get_all_qualities(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        tracker.register(1)
        tracker.register(2)
        tracker.record_reconnect(1)

        all_q = tracker.get_all_qualities()
        assert 1 in all_q
        assert 2 in all_q
        assert all_q[1]["reconnect_count"] == 1
        assert all_q[2]["reconnect_count"] == 0

    def test_nonexistent_client_returns_none(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        assert tracker.get_quality(999) is None

    def test_operations_on_nonexistent_client_are_safe(self):
        from aragora.server.stream.replay_buffer import ConnectionQualityTracker

        tracker = ConnectionQualityTracker()
        # These should not raise
        tracker.record_reconnect(999)
        tracker.record_replay(999, 10)
        tracker.record_message_sent(999)
        tracker.update_last_seen_seq(999, 50)
        tracker.record_latency(999, 10.0)
        assert tracker.unregister(999) is None
