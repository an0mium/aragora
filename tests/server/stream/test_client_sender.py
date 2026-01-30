"""
Tests for timeout-safe WebSocket client sender with health tracking.

Tests cover:
- TimeoutSender: Timeout-safe WebSocket sending
- ClientHealth: Health tracking and metrics
- Quarantine logic for failing clients
- Retry behavior with backoff
- Connection state management
- Concurrent message handling
- Error handling for closed connections
- Global sender instance management
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.stream.client_sender import (
    ClientHealth,
    ClientStatus,
    TimeoutSender,
    get_timeout_sender,
    reset_timeout_sender,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def sender():
    """Create a TimeoutSender with test-friendly config."""
    return TimeoutSender(
        timeout=0.5,  # Short timeout for faster tests
        max_failures=3,
        quarantine_duration=1.0,
    )


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket with send_str method (aiohttp style)."""
    ws = MagicMock()
    ws.send_str = AsyncMock()
    return ws


@pytest.fixture
def mock_websocket_send():
    """Create a mock WebSocket with send method only (websockets style)."""
    # Use spec to ensure send_str doesn't exist
    ws = MagicMock(spec=["send"])
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def slow_websocket():
    """Create a mock WebSocket that delays on send."""
    ws = MagicMock()

    async def slow_send(msg):
        await asyncio.sleep(2.0)  # Longer than typical timeout

    ws.send_str = AsyncMock(side_effect=slow_send)
    return ws


@pytest.fixture
def failing_websocket():
    """Create a mock WebSocket that raises errors on send."""
    ws = MagicMock()
    ws.send_str = AsyncMock(side_effect=ConnectionError("Connection closed"))
    return ws


@pytest.fixture(autouse=True)
def reset_global_sender():
    """Reset the global sender before and after each test."""
    reset_timeout_sender()
    yield
    reset_timeout_sender()


# ===========================================================================
# Test ClientStatus Enum
# ===========================================================================


class TestClientStatus:
    """Tests for ClientStatus enum."""

    def test_status_values(self):
        """ClientStatus has expected values."""
        assert ClientStatus.HEALTHY.value == "healthy"
        assert ClientStatus.DEGRADED.value == "degraded"
        assert ClientStatus.QUARANTINED.value == "quarantined"

    def test_status_comparison(self):
        """Status values can be compared."""
        assert ClientStatus.HEALTHY != ClientStatus.DEGRADED
        assert ClientStatus.DEGRADED != ClientStatus.QUARANTINED


# ===========================================================================
# Test ClientHealth Dataclass
# ===========================================================================


class TestClientHealth:
    """Tests for ClientHealth dataclass."""

    def test_initial_state(self):
        """New ClientHealth has healthy initial state."""
        health = ClientHealth(client_id=123)
        assert health.client_id == 123
        assert health.consecutive_failures == 0
        assert health.total_failures == 0
        assert health.total_sends == 0
        assert health.last_failure_time is None
        assert health.last_success_time is None
        assert health.quarantined_until is None
        assert health.avg_latency_ms == 0.0
        assert health.status == ClientStatus.HEALTHY

    def test_status_healthy(self):
        """Status is HEALTHY when no failures."""
        health = ClientHealth(client_id=1)
        assert health.status == ClientStatus.HEALTHY

    def test_status_degraded_on_failures(self):
        """Status becomes DEGRADED after 2+ consecutive failures."""
        health = ClientHealth(client_id=1)
        health.consecutive_failures = 2
        assert health.status == ClientStatus.DEGRADED

    def test_status_quarantined_when_set(self):
        """Status is QUARANTINED when quarantine is active."""
        health = ClientHealth(client_id=1)
        health.quarantined_until = time.time() + 10.0
        assert health.status == ClientStatus.QUARANTINED

    def test_status_healthy_after_quarantine_expires(self):
        """Status returns to HEALTHY after quarantine expires."""
        health = ClientHealth(client_id=1)
        health.quarantined_until = time.time() - 1.0  # Expired
        assert health.status == ClientStatus.HEALTHY

    def test_failure_rate_zero_when_no_sends(self):
        """Failure rate is 0 when no sends attempted."""
        health = ClientHealth(client_id=1)
        assert health.failure_rate == 0.0

    def test_failure_rate_calculation(self):
        """Failure rate calculates correctly."""
        health = ClientHealth(client_id=1)
        health.total_sends = 10
        health.total_failures = 2
        assert health.failure_rate == 20.0

    def test_record_success_updates_metrics(self):
        """record_success updates all relevant metrics."""
        health = ClientHealth(client_id=1)
        health.consecutive_failures = 3  # Simulate prior failures

        health.record_success(latency_ms=50.0)

        assert health.consecutive_failures == 0
        assert health.total_sends == 1
        assert health.last_success_time is not None
        assert health.avg_latency_ms == 50.0

    def test_record_success_clears_quarantine(self):
        """Successful send clears any active quarantine."""
        health = ClientHealth(client_id=1)
        health.quarantined_until = time.time() + 100.0

        health.record_success(latency_ms=10.0)

        assert health.quarantined_until is None
        assert health.status == ClientStatus.HEALTHY

    def test_record_success_rolling_average_latency(self):
        """Latency tracking maintains rolling average of last 10 samples."""
        health = ClientHealth(client_id=1)

        # Add 15 samples
        for i in range(15):
            health.record_success(latency_ms=float(i * 10))

        # Average should be based on last 10 samples (50 to 140)
        expected_avg = sum(range(5, 15)) * 10 / 10
        assert health.avg_latency_ms == expected_avg

    def test_record_failure_increments_counters(self):
        """record_failure increments failure counters."""
        health = ClientHealth(client_id=1)

        health.record_failure()

        assert health.consecutive_failures == 1
        assert health.total_failures == 1
        assert health.total_sends == 1
        assert health.last_failure_time is not None

    def test_record_failure_with_quarantine(self):
        """record_failure can set quarantine duration."""
        health = ClientHealth(client_id=1)

        health.record_failure(quarantine_duration=30.0)

        assert health.quarantined_until is not None
        assert health.quarantined_until > time.time()
        assert health.status == ClientStatus.QUARANTINED


# ===========================================================================
# Test TimeoutSender Initialization
# ===========================================================================


class TestTimeoutSenderInit:
    """Tests for TimeoutSender initialization."""

    def test_default_config(self):
        """Default configuration has sensible defaults."""
        sender = TimeoutSender()
        assert sender._timeout == 2.0
        assert sender._max_failures == 3
        assert sender._quarantine_duration == 10.0

    def test_custom_config(self):
        """Custom configuration is respected."""
        sender = TimeoutSender(
            timeout=5.0,
            max_failures=5,
            quarantine_duration=60.0,
        )
        assert sender._timeout == 5.0
        assert sender._max_failures == 5
        assert sender._quarantine_duration == 60.0

    def test_initial_stats_are_zero(self):
        """Initial statistics are all zero."""
        sender = TimeoutSender()
        stats = sender.get_stats()
        assert stats["total_sends"] == 0
        assert stats["total_failures"] == 0
        assert stats["total_timeouts"] == 0
        assert stats["total_quarantines"] == 0


# ===========================================================================
# Test TimeoutSender.send()
# ===========================================================================


class TestTimeoutSenderSend:
    """Tests for TimeoutSender.send() method."""

    @pytest.mark.asyncio
    async def test_send_success_with_send_str(self, sender, mock_websocket):
        """Successful send using aiohttp-style send_str."""
        result = await sender.send(mock_websocket, "Hello")

        assert result is True
        mock_websocket.send_str.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_send_success_with_send(self, sender, mock_websocket_send):
        """Successful send using websockets-style send."""
        result = await sender.send(mock_websocket_send, "Hello")

        assert result is True
        mock_websocket_send.send.assert_called_once_with("Hello")

    @pytest.mark.asyncio
    async def test_send_records_health(self, sender, mock_websocket):
        """Successful send records health metrics."""
        await sender.send(mock_websocket, "Hello")

        health = sender.get_health(mock_websocket)
        assert health is not None
        assert health.total_sends == 1
        assert health.total_failures == 0
        assert health.status == ClientStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_send_timeout_returns_false(self, sender, slow_websocket):
        """Send that times out returns False."""
        result = await sender.send(slow_websocket, "Hello", timeout=0.1)

        assert result is False

    @pytest.mark.asyncio
    async def test_send_timeout_records_failure(self, sender, slow_websocket):
        """Timeout records as failure in health metrics."""
        await sender.send(slow_websocket, "Hello", timeout=0.1)

        health = sender.get_health(slow_websocket)
        assert health.total_failures == 1
        assert health.consecutive_failures == 1
        assert sender._total_timeouts == 1

    @pytest.mark.asyncio
    async def test_send_connection_error_returns_false(self, sender, failing_websocket):
        """Send that raises ConnectionError returns False."""
        result = await sender.send(failing_websocket, "Hello")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_os_error_returns_false(self, sender):
        """Send that raises OSError returns False."""
        ws = MagicMock()
        ws.send_str = AsyncMock(side_effect=OSError("Broken pipe"))

        result = await sender.send(ws, "Hello")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_runtime_error_returns_false(self, sender):
        """Send that raises RuntimeError returns False."""
        ws = MagicMock()
        ws.send_str = AsyncMock(side_effect=RuntimeError("WebSocket closed"))

        result = await sender.send(ws, "Hello")

        assert result is False

    @pytest.mark.asyncio
    async def test_send_unsupported_client_type_raises(self, sender):
        """Unsupported WebSocket client type raises TypeError."""
        ws = MagicMock(spec=[])  # No send or send_str methods

        with pytest.raises(TypeError, match="Unsupported WebSocket client type"):
            await sender.send(ws, "Hello")

    @pytest.mark.asyncio
    async def test_send_custom_timeout_override(self, sender, slow_websocket):
        """Custom timeout parameter overrides default."""
        # Default is 0.5s, slow_websocket delays 2s
        # This should timeout with 0.1s override
        result = await sender.send(slow_websocket, "Hello", timeout=0.1)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_skips_quarantined_client(self, sender, mock_websocket):
        """Quarantined clients are skipped without attempting send."""
        # Manually quarantine the client
        client_id = id(mock_websocket)
        health = sender._get_or_create_health(client_id)
        health.quarantined_until = time.time() + 100.0

        result = await sender.send(mock_websocket, "Hello")

        assert result is False
        mock_websocket.send_str.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_increments_total_sends_counter(self, sender, mock_websocket):
        """Each send attempt increments total_sends counter."""
        await sender.send(mock_websocket, "Hello")
        await sender.send(mock_websocket, "World")

        assert sender._total_sends == 2


# ===========================================================================
# Test Quarantine Logic
# ===========================================================================


class TestQuarantineLogic:
    """Tests for quarantine logic."""

    @pytest.mark.asyncio
    async def test_client_quarantined_after_max_failures(self, sender, failing_websocket):
        """Client is quarantined after exceeding max_failures.

        With max_failures=3, quarantine triggers on the 4th failure because:
        - _handle_failure checks consecutive_failures >= max_failures BEFORE incrementing
        - So failures 1-3 increment without triggering, failure 4 triggers quarantine
        """
        # max_failures is 3, so 4 failures trigger quarantine
        for _ in range(4):
            await sender.send(failing_websocket, "Hello")

        assert sender.is_quarantined(failing_websocket) is True
        assert sender._total_quarantines == 1

    @pytest.mark.asyncio
    async def test_quarantine_duration_respected(self, failing_websocket):
        """Quarantine lasts for configured duration."""
        sender = TimeoutSender(timeout=0.5, max_failures=2, quarantine_duration=0.5)

        # Trigger quarantine (needs max_failures + 1 = 3 failures)
        for _ in range(3):
            await sender.send(failing_websocket, "Hello")

        assert sender.is_quarantined(failing_websocket) is True

        # Wait for quarantine to expire
        await asyncio.sleep(0.6)

        assert sender.is_quarantined(failing_websocket) is False

    @pytest.mark.asyncio
    async def test_success_clears_consecutive_failures(self, sender, mock_websocket):
        """Successful send resets consecutive failure count."""
        # Record some failures manually
        health = sender._get_or_create_health(id(mock_websocket))
        health.consecutive_failures = 2

        # Successful send should reset
        await sender.send(mock_websocket, "Hello")

        assert health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_quarantine_cleared_on_success(self, sender, mock_websocket):
        """Quarantine is cleared when client succeeds."""
        # Manually set quarantine
        health = sender._get_or_create_health(id(mock_websocket))
        health.quarantined_until = time.time() - 0.1  # Just expired

        # Now it should work
        result = await sender.send(mock_websocket, "Hello")
        assert result is True
        assert health.quarantined_until is None


# ===========================================================================
# Test TimeoutSender.send_many()
# ===========================================================================


class TestTimeoutSenderSendMany:
    """Tests for TimeoutSender.send_many() method."""

    @pytest.mark.asyncio
    async def test_send_many_empty_list(self, sender):
        """Empty client list returns (0, [])."""
        success_count, dead_clients = await sender.send_many([], "Hello")

        assert success_count == 0
        assert dead_clients == []

    @pytest.mark.asyncio
    async def test_send_many_all_succeed(self, sender):
        """All successful sends returns correct count."""
        clients = [MagicMock() for _ in range(3)]
        for c in clients:
            c.send_str = AsyncMock()

        success_count, dead_clients = await sender.send_many(clients, "Hello")

        assert success_count == 3
        assert dead_clients == []

    @pytest.mark.asyncio
    async def test_send_many_some_fail(self, sender):
        """Mixed success/failure reports correctly."""
        good_client = MagicMock()
        good_client.send_str = AsyncMock()

        bad_client = MagicMock()
        bad_client.send_str = AsyncMock(side_effect=ConnectionError())

        success_count, dead_clients = await sender.send_many([good_client, bad_client], "Hello")

        assert success_count == 1
        # bad_client only has 1 failure, not dead yet
        assert bad_client not in dead_clients

    @pytest.mark.asyncio
    async def test_send_many_identifies_dead_clients(self, sender):
        """Dead clients (max failures) are returned in list."""
        bad_client = MagicMock()
        bad_client.send_str = AsyncMock(side_effect=ConnectionError())

        # Pre-fail the client to get close to threshold
        health = sender._get_or_create_health(id(bad_client))
        health.consecutive_failures = 2

        success_count, dead_clients = await sender.send_many([bad_client], "Hello")

        assert success_count == 0
        assert bad_client in dead_clients

    @pytest.mark.asyncio
    async def test_send_many_concurrent_sends(self, sender):
        """send_many sends to all clients concurrently."""
        call_times = []

        async def record_call_time(msg):
            call_times.append(time.time())
            await asyncio.sleep(0.1)

        clients = [MagicMock() for _ in range(3)]
        for c in clients:
            c.send_str = AsyncMock(side_effect=record_call_time)

        start = time.time()
        await sender.send_many(clients, "Hello")
        duration = time.time() - start

        # If concurrent, total time should be ~0.1s not ~0.3s
        assert duration < 0.2

    @pytest.mark.asyncio
    async def test_send_many_skips_quarantined(self, sender):
        """Quarantined clients are skipped in send_many."""
        good_client = MagicMock()
        good_client.send_str = AsyncMock()

        quarantined_client = MagicMock()
        quarantined_client.send_str = AsyncMock()

        # Quarantine one client
        health = sender._get_or_create_health(id(quarantined_client))
        health.quarantined_until = time.time() + 100.0

        success_count, _ = await sender.send_many([good_client, quarantined_client], "Hello")

        assert success_count == 1
        quarantined_client.send_str.assert_not_called()


# ===========================================================================
# Test Health Checking Methods
# ===========================================================================


class TestHealthChecking:
    """Tests for health checking methods."""

    def test_is_quarantined_unknown_client(self, sender):
        """is_quarantined returns False for unknown client."""
        ws = MagicMock()
        assert sender.is_quarantined(ws) is False

    def test_is_quarantined_healthy_client(self, sender, mock_websocket):
        """is_quarantined returns False for healthy client."""
        sender._get_or_create_health(id(mock_websocket))
        assert sender.is_quarantined(mock_websocket) is False

    def test_is_dead_unknown_client(self, sender):
        """is_dead returns False for unknown client."""
        ws = MagicMock()
        assert sender.is_dead(ws) is False

    def test_is_dead_healthy_client(self, sender, mock_websocket):
        """is_dead returns False for healthy client."""
        sender._get_or_create_health(id(mock_websocket))
        assert sender.is_dead(mock_websocket) is False

    def test_is_dead_after_max_failures(self, sender, mock_websocket):
        """is_dead returns True after max_failures."""
        health = sender._get_or_create_health(id(mock_websocket))
        health.consecutive_failures = 3  # Equal to max_failures

        assert sender.is_dead(mock_websocket) is True

    def test_get_health_unknown_client(self, sender):
        """get_health returns None for unknown client."""
        ws = MagicMock()
        assert sender.get_health(ws) is None

    def test_get_health_tracked_client(self, sender, mock_websocket):
        """get_health returns ClientHealth for tracked client."""
        sender._get_or_create_health(id(mock_websocket))
        health = sender.get_health(mock_websocket)

        assert health is not None
        assert isinstance(health, ClientHealth)


# ===========================================================================
# Test Client Management
# ===========================================================================


class TestClientManagement:
    """Tests for client management methods."""

    def test_remove_client(self, sender, mock_websocket):
        """remove_client removes health tracking for client."""
        sender._get_or_create_health(id(mock_websocket))
        assert sender.get_health(mock_websocket) is not None

        sender.remove_client(mock_websocket)

        assert sender.get_health(mock_websocket) is None

    def test_remove_client_unknown(self, sender):
        """remove_client handles unknown client gracefully."""
        ws = MagicMock()
        sender.remove_client(ws)  # Should not raise


# ===========================================================================
# Test Statistics
# ===========================================================================


class TestStatistics:
    """Tests for get_stats method."""

    def test_get_stats_initial(self, sender):
        """Initial stats are all zero."""
        stats = sender.get_stats()

        assert stats["total_sends"] == 0
        assert stats["total_failures"] == 0
        assert stats["total_timeouts"] == 0
        assert stats["total_quarantines"] == 0
        assert stats["tracked_clients"] == 0
        assert stats["quarantined_clients"] == 0
        assert stats["degraded_clients"] == 0
        assert stats["failure_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_after_sends(self, sender, mock_websocket, failing_websocket):
        """Stats update after sends."""
        await sender.send(mock_websocket, "Hello")
        await sender.send(failing_websocket, "Hello")

        stats = sender.get_stats()

        assert stats["total_sends"] == 2
        assert stats["total_failures"] == 1
        assert stats["tracked_clients"] == 2
        assert stats["failure_rate"] == 50.0

    @pytest.mark.asyncio
    async def test_get_stats_counts_degraded(self, sender, failing_websocket):
        """Stats count degraded clients."""
        # 2 failures makes client degraded
        await sender.send(failing_websocket, "Hello")
        await sender.send(failing_websocket, "Hello")

        stats = sender.get_stats()

        assert stats["degraded_clients"] == 1

    @pytest.mark.asyncio
    async def test_get_stats_counts_quarantined(self, sender, failing_websocket):
        """Stats count quarantined clients."""
        # With max_failures=3, quarantine triggers on 4th failure
        for _ in range(4):
            await sender.send(failing_websocket, "Hello")

        stats = sender.get_stats()

        assert stats["quarantined_clients"] == 1
        assert stats["total_quarantines"] == 1


# ===========================================================================
# Test Global Sender Instance
# ===========================================================================


class TestGlobalSender:
    """Tests for global sender instance management."""

    def test_get_timeout_sender_creates_instance(self):
        """get_timeout_sender creates a new instance if none exists."""
        sender = get_timeout_sender()
        assert sender is not None
        assert isinstance(sender, TimeoutSender)

    def test_get_timeout_sender_returns_same_instance(self):
        """get_timeout_sender returns the same instance on repeated calls."""
        sender1 = get_timeout_sender()
        sender2 = get_timeout_sender()
        assert sender1 is sender2

    def test_get_timeout_sender_with_config(self):
        """get_timeout_sender accepts configuration on first call."""
        sender = get_timeout_sender(timeout=5.0, max_failures=10, quarantine_duration=30.0)
        assert sender._timeout == 5.0
        assert sender._max_failures == 10
        assert sender._quarantine_duration == 30.0

    def test_get_timeout_sender_ignores_config_after_first_call(self):
        """Configuration is ignored after first call."""
        sender1 = get_timeout_sender(timeout=5.0)
        sender2 = get_timeout_sender(timeout=10.0)

        assert sender2._timeout == 5.0  # Still uses first config

    def test_reset_timeout_sender(self):
        """reset_timeout_sender clears the global instance."""
        sender1 = get_timeout_sender()
        reset_timeout_sender()
        sender2 = get_timeout_sender()

        assert sender1 is not sender2


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_multiple_clients_independent_health(self, sender):
        """Each client has independent health tracking."""
        ws1 = MagicMock()
        ws1.send_str = AsyncMock()

        ws2 = MagicMock()
        ws2.send_str = AsyncMock(side_effect=ConnectionError())

        await sender.send(ws1, "Hello")
        await sender.send(ws2, "Hello")

        health1 = sender.get_health(ws1)
        health2 = sender.get_health(ws2)

        assert health1.total_failures == 0
        assert health2.total_failures == 1

    @pytest.mark.asyncio
    async def test_very_fast_send_records_low_latency(self, sender, mock_websocket):
        """Fast sends record low latency."""
        await sender.send(mock_websocket, "Hello")

        health = sender.get_health(mock_websocket)
        assert health.avg_latency_ms < 100  # Should be very fast

    @pytest.mark.asyncio
    async def test_send_empty_message(self, sender, mock_websocket):
        """Empty message can be sent."""
        result = await sender.send(mock_websocket, "")
        assert result is True
        mock_websocket.send_str.assert_called_once_with("")

    @pytest.mark.asyncio
    async def test_send_large_message(self, sender, mock_websocket):
        """Large message can be sent."""
        large_msg = "x" * 100000
        result = await sender.send(mock_websocket, large_msg)
        assert result is True
        mock_websocket.send_str.assert_called_once_with(large_msg)

    @pytest.mark.asyncio
    async def test_rapid_sequential_sends(self, sender, mock_websocket):
        """Rapid sequential sends all succeed."""
        for i in range(100):
            result = await sender.send(mock_websocket, f"Message {i}")
            assert result is True

        health = sender.get_health(mock_websocket)
        assert health.total_sends == 100
        assert health.total_failures == 0

    @pytest.mark.asyncio
    async def test_zero_timeout_immediate_failure(self, sender, slow_websocket):
        """Zero timeout causes immediate failure."""
        result = await sender.send(slow_websocket, "Hello", timeout=0.001)
        assert result is False

    @pytest.mark.asyncio
    async def test_failure_recovery_pattern(self, sender):
        """Client can recover from failures with successful sends."""
        ws = MagicMock()
        call_count = [0]

        async def sometimes_fail(msg):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ConnectionError()

        ws.send_str = AsyncMock(side_effect=sometimes_fail)

        # First two fail
        await sender.send(ws, "Hello")
        await sender.send(ws, "Hello")

        health = sender.get_health(ws)
        assert health.consecutive_failures == 2

        # Third succeeds
        await sender.send(ws, "Hello")
        assert health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_concurrent_sends_same_client(self, sender, mock_websocket):
        """Concurrent sends to same client all tracked correctly."""

        # Create slight delay to ensure overlap
        async def delayed_send(msg):
            await asyncio.sleep(0.01)

        mock_websocket.send_str = AsyncMock(side_effect=delayed_send)

        tasks = [sender.send(mock_websocket, f"Message {i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert all(results)
        health = sender.get_health(mock_websocket)
        assert health.total_sends == 10


# ===========================================================================
# Test Thread Safety (Lock)
# ===========================================================================


class TestThreadSafety:
    """Tests for thread safety with the internal lock."""

    @pytest.mark.asyncio
    async def test_concurrent_health_updates(self, sender):
        """Concurrent sends don't corrupt health data."""
        clients = [MagicMock() for _ in range(5)]
        for c in clients:
            c.send_str = AsyncMock()

        # Send many messages concurrently
        tasks = []
        for _ in range(20):
            for c in clients:
                tasks.append(sender.send(c, "Hello"))

        await asyncio.gather(*tasks)

        # Verify consistent state
        stats = sender.get_stats()
        assert stats["total_sends"] == 100
        assert stats["tracked_clients"] == 5

        for c in clients:
            health = sender.get_health(c)
            assert health.total_sends == 20


# ===========================================================================
# Test Latency Tracking
# ===========================================================================


class TestLatencyTracking:
    """Tests for latency tracking in ClientHealth."""

    def test_latency_samples_limited_to_10(self):
        """Latency samples are limited to last 10."""
        health = ClientHealth(client_id=1)

        for i in range(20):
            health.record_success(latency_ms=float(i))

        assert len(health._latency_samples) == 10
        assert health._latency_samples[0] == 10.0  # Oldest of last 10
        assert health._latency_samples[-1] == 19.0  # Most recent

    def test_avg_latency_rolling_window(self):
        """Average latency uses rolling window."""
        health = ClientHealth(client_id=1)

        # Add exactly 10 samples with known values
        for i in range(10):
            health.record_success(latency_ms=100.0)

        assert health.avg_latency_ms == 100.0

        # Add one more, pushing out old one
        health.record_success(latency_ms=200.0)

        # New avg: (9 * 100 + 200) / 10 = 110
        assert health.avg_latency_ms == 110.0
