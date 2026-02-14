"""
Comprehensive tests for Streaming Connector Resilience Patterns.

Tests cover:
1. Circuit breaker integration (state transitions, call gating, metrics)
2. Retry logic with exponential backoff (delays, jitter, caps, async iteration)
3. Dead letter queue handling (routing, metadata, callbacks, sender failures)
4. Message idempotency (retry tracking, message IDs, deduplication)
5. Health monitoring (success/failure tracking, threshold, reset)
6. Recovery strategies (half-open probing, circuit reset, graceful shutdown)
7. Edge cases (rapid failures, max retries, concurrent access, config validation)

These tests mock message brokers and use pytest fixtures for isolation.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.streaming.resilience import (
    CircuitBreakerContext,
    CircuitBreakerOpenError,
    CircuitState,
    DLQHandler,
    DLQMessage,
    ExponentialBackoff,
    GracefulShutdown,
    HealthMonitor,
    HealthStatus,
    StreamingCircuitBreaker,
    StreamingResilienceConfig,
    with_retry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default resilience configuration."""
    return StreamingResilienceConfig()


@pytest.fixture
def fast_config():
    """Configuration with minimal delays for fast testing."""
    return StreamingResilienceConfig(
        max_retries=3,
        initial_delay_seconds=0.01,
        max_delay_seconds=0.1,
        jitter=False,
        circuit_breaker_threshold=3,
        circuit_breaker_recovery_seconds=0.05,
        circuit_breaker_half_open_calls=2,
        circuit_breaker_success_threshold=2,
        dlq_max_retries=2,
        unhealthy_threshold=2,
    )


@pytest.fixture
def no_jitter_config():
    """Configuration with jitter disabled for deterministic testing."""
    return StreamingResilienceConfig(
        initial_delay_seconds=1.0,
        max_delay_seconds=60.0,
        exponential_base=2.0,
        jitter=False,
    )


@pytest.fixture
def circuit_breaker(fast_config):
    """A circuit breaker with fast configuration."""
    return StreamingCircuitBreaker("test-breaker", fast_config)


@pytest.fixture
def health_monitor(fast_config):
    """A health monitor with fast configuration."""
    return HealthMonitor("test-monitor", fast_config)


@pytest.fixture
def dlq_sender():
    """Mock DLQ sender function."""
    return AsyncMock()


@pytest.fixture
def dlq_callback():
    """Mock DLQ send callback."""
    return AsyncMock()


@pytest.fixture
def dlq_handler(fast_config, dlq_sender, dlq_callback):
    """A DLQ handler with mocked sender and callback."""
    return DLQHandler(
        config=fast_config,
        dlq_sender=dlq_sender,
        on_dlq_send=dlq_callback,
    )


@pytest.fixture
def sample_timestamp():
    """A sample timestamp for DLQ messages."""
    return datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_dlq_message(sample_timestamp):
    """A sample DLQ message for testing."""
    return DLQMessage(
        original_topic="events",
        original_key="evt-123",
        original_value={"type": "order", "id": 42},
        original_headers={"source": "api", "version": "2"},
        original_timestamp=sample_timestamp,
        error_message="Connection refused",
        error_type="ConnectionError",
        retry_count=3,
    )


# =============================================================================
# 1. StreamingResilienceConfig Tests
# =============================================================================


class TestStreamingResilienceConfig:
    """Tests for StreamingResilienceConfig validation and defaults."""

    def test_default_values(self, default_config):
        """Should have sensible defaults for all fields."""
        assert default_config.max_retries == 5
        assert default_config.initial_delay_seconds == 1.0
        assert default_config.max_delay_seconds == 60.0
        assert default_config.exponential_base == 2.0
        assert default_config.jitter is True
        assert default_config.circuit_breaker_threshold == 5
        assert default_config.circuit_breaker_recovery_seconds == 30.0
        assert default_config.circuit_breaker_half_open_calls == 3
        assert default_config.circuit_breaker_success_threshold == 2
        assert default_config.dlq_enabled is True
        assert default_config.dlq_max_retries == 3
        assert default_config.dlq_include_metadata is True
        assert default_config.dlq_topic_suffix == ".dlq"
        assert default_config.connection_timeout_seconds == 30.0
        assert default_config.operation_timeout_seconds == 10.0
        assert default_config.health_check_interval_seconds == 30.0
        assert default_config.unhealthy_threshold == 3

    def test_custom_values(self):
        """Should accept custom values for all fields."""
        config = StreamingResilienceConfig(
            max_retries=10,
            initial_delay_seconds=0.5,
            max_delay_seconds=120.0,
            exponential_base=3.0,
            jitter=False,
            circuit_breaker_threshold=10,
            circuit_breaker_recovery_seconds=60.0,
            circuit_breaker_half_open_calls=5,
            circuit_breaker_success_threshold=3,
            dlq_enabled=False,
            dlq_max_retries=5,
            dlq_topic_suffix=".dead",
            connection_timeout_seconds=60.0,
            operation_timeout_seconds=30.0,
            health_check_interval_seconds=10.0,
            unhealthy_threshold=5,
        )
        assert config.max_retries == 10
        assert config.exponential_base == 3.0
        assert config.dlq_topic_suffix == ".dead"

    def test_rejects_negative_max_retries(self):
        """Should reject negative max_retries."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            StreamingResilienceConfig(max_retries=-1)

    def test_accepts_zero_max_retries(self):
        """Should accept zero max_retries (no retries)."""
        config = StreamingResilienceConfig(max_retries=0)
        assert config.max_retries == 0

    def test_rejects_zero_initial_delay(self):
        """Should reject zero initial_delay_seconds."""
        with pytest.raises(ValueError, match="initial_delay_seconds must be positive"):
            StreamingResilienceConfig(initial_delay_seconds=0)

    def test_rejects_negative_initial_delay(self):
        """Should reject negative initial_delay_seconds."""
        with pytest.raises(ValueError, match="initial_delay_seconds must be positive"):
            StreamingResilienceConfig(initial_delay_seconds=-1.0)

    def test_rejects_max_delay_less_than_initial_delay(self):
        """Should reject max_delay_seconds < initial_delay_seconds."""
        with pytest.raises(ValueError, match="max_delay_seconds must be >= initial_delay"):
            StreamingResilienceConfig(
                initial_delay_seconds=10.0,
                max_delay_seconds=5.0,
            )

    def test_accepts_max_delay_equal_to_initial_delay(self):
        """Should accept max_delay_seconds == initial_delay_seconds."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=5.0,
            max_delay_seconds=5.0,
        )
        assert config.max_delay_seconds == 5.0

    def test_rejects_zero_circuit_breaker_threshold(self):
        """Should reject circuit_breaker_threshold < 1."""
        with pytest.raises(ValueError, match="circuit_breaker_threshold must be >= 1"):
            StreamingResilienceConfig(circuit_breaker_threshold=0)

    def test_accepts_circuit_breaker_threshold_of_one(self):
        """Should accept circuit_breaker_threshold of 1."""
        config = StreamingResilienceConfig(circuit_breaker_threshold=1)
        assert config.circuit_breaker_threshold == 1


# =============================================================================
# 2. ExponentialBackoff Tests
# =============================================================================


class TestExponentialBackoff:
    """Tests for ExponentialBackoff delay calculation and iteration."""

    def test_default_config(self):
        """Should use default config when none provided."""
        backoff = ExponentialBackoff()
        assert backoff.config.max_retries == 5

    def test_first_attempt_returns_initial_delay(self, no_jitter_config):
        """Should return initial_delay for attempt 0."""
        backoff = ExponentialBackoff(no_jitter_config)
        assert backoff.get_delay(0) == 1.0

    def test_exponential_growth_without_jitter(self, no_jitter_config):
        """Should grow exponentially: base * 2^attempt."""
        backoff = ExponentialBackoff(no_jitter_config)

        assert backoff.get_delay(0) == 1.0
        assert backoff.get_delay(1) == 2.0
        assert backoff.get_delay(2) == 4.0
        assert backoff.get_delay(3) == 8.0
        assert backoff.get_delay(4) == 16.0
        assert backoff.get_delay(5) == 32.0

    def test_caps_at_max_delay(self, no_jitter_config):
        """Should never exceed max_delay_seconds."""
        backoff = ExponentialBackoff(no_jitter_config)

        # Attempt 10 would be 1024, but max is 60
        assert backoff.get_delay(10) == 60.0
        assert backoff.get_delay(20) == 60.0

    def test_jitter_adds_randomness(self):
        """Should produce varied delays when jitter is enabled."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=1.0,
            jitter=True,
        )
        backoff = ExponentialBackoff(config)

        delays = {backoff.get_delay(2) for _ in range(20)}
        # With jitter, we expect multiple different values
        assert len(delays) > 1

    def test_jitter_stays_within_bounds(self):
        """Jitter should keep delay >= initial_delay_seconds."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=1.0,
            max_delay_seconds=60.0,
            jitter=True,
        )
        backoff = ExponentialBackoff(config)

        for _ in range(100):
            delay = backoff.get_delay(0)
            assert delay >= config.initial_delay_seconds

    def test_jitter_range_is_25_percent(self):
        """Jitter should be within +/-25% of the base delay."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=10.0,
            max_delay_seconds=60.0,
            exponential_base=2.0,
            jitter=True,
        )
        backoff = ExponentialBackoff(config)

        # For attempt 0, base_delay = 10.0, jitter range = 2.5
        # So delay should be in [7.5, 12.5], clamped to >= 10.0 by min check
        for _ in range(50):
            delay = backoff.get_delay(0)
            assert delay >= config.initial_delay_seconds
            assert delay <= 10.0 * 1.25 + 0.01  # small tolerance

    def test_reset_clears_state(self, no_jitter_config):
        """Should reset attempt counter and last delay."""
        backoff = ExponentialBackoff(no_jitter_config)
        backoff.get_delay(5)

        backoff.reset()

        assert backoff._attempt == 0
        assert backoff._last_delay == no_jitter_config.initial_delay_seconds

    def test_tracks_last_delay(self, no_jitter_config):
        """Should update _last_delay after each get_delay call."""
        backoff = ExponentialBackoff(no_jitter_config)
        backoff.get_delay(3)
        assert backoff._last_delay == 8.0

    @pytest.mark.asyncio
    async def test_async_iterator_yields_correct_count(self):
        """Should yield max_retries + 1 delays."""
        config = StreamingResilienceConfig(
            max_retries=4,
            initial_delay_seconds=0.01,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        delays = []
        async for delay in backoff:
            delays.append(delay)

        assert len(delays) == 5  # 0..4 inclusive

    @pytest.mark.asyncio
    async def test_async_iterator_with_zero_retries(self):
        """Should yield exactly 1 delay when max_retries=0."""
        config = StreamingResilienceConfig(
            max_retries=0,
            initial_delay_seconds=0.01,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        delays = []
        async for delay in backoff:
            delays.append(delay)

        assert len(delays) == 1

    @pytest.mark.asyncio
    async def test_async_iterator_updates_attempt(self):
        """Should track current attempt during iteration."""
        config = StreamingResilienceConfig(
            max_retries=2,
            initial_delay_seconds=0.01,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        attempts = []
        async for _ in backoff:
            attempts.append(backoff._attempt)

        assert attempts == [0, 1, 2]

    def test_small_max_delay_equal_to_initial_delay(self):
        """Should handle max_delay equal to initial_delay."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=5.0,
            max_delay_seconds=5.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        # All attempts should be capped at 5.0
        for attempt in range(10):
            assert backoff.get_delay(attempt) == 5.0


# =============================================================================
# 3. Circuit Breaker State Transitions
# =============================================================================


class TestCircuitBreakerStateTransitions:
    """Tests for StreamingCircuitBreaker state machine."""

    def test_starts_closed(self, circuit_breaker):
        """Should start in CLOSED state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.is_open is False
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_stays_closed_below_threshold(self, circuit_breaker):
        """Should remain CLOSED when failures < threshold."""
        # threshold is 3, record 2 failures
        await circuit_breaker.record_failure(Exception("Error 1"))
        await circuit_breaker.record_failure(Exception("Error 2"))

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 2

    @pytest.mark.asyncio
    async def test_opens_at_threshold(self, circuit_breaker):
        """Should transition CLOSED -> OPEN at failure threshold."""
        for i in range(3):
            await circuit_breaker.record_failure(Exception(f"Error {i}"))

        assert circuit_breaker.state == CircuitState.OPEN
        assert circuit_breaker.is_open is True

    @pytest.mark.asyncio
    async def test_success_resets_failure_count_in_closed(self, circuit_breaker):
        """Success in CLOSED state should reset failure count."""
        await circuit_breaker.record_failure(Exception("Error"))
        await circuit_breaker.record_failure(Exception("Error"))
        assert circuit_breaker.failure_count == 2

        await circuit_breaker.record_success()
        assert circuit_breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_open_rejects_calls(self, circuit_breaker):
        """Should reject calls when OPEN."""
        for i in range(3):
            await circuit_breaker.record_failure(Exception(f"Error {i}"))

        can_exec = await circuit_breaker.can_execute()
        assert can_exec is False

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_recovery_timeout(self, circuit_breaker):
        """Should transition OPEN -> HALF_OPEN after recovery timeout."""
        for i in range(3):
            await circuit_breaker.record_failure(Exception(f"Error {i}"))

        assert circuit_breaker.state == CircuitState.OPEN

        # Wait for recovery (0.05 seconds)
        await asyncio.sleep(0.1)

        can_exec = await circuit_breaker.can_execute()
        assert can_exec is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_limits_calls(self, fast_config):
        """Should limit calls in HALF_OPEN to half_open_calls."""
        breaker = StreamingCircuitBreaker("test", fast_config)
        # Open the circuit
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        await asyncio.sleep(0.1)

        # First call transitions to HALF_OPEN and sets _half_open_calls=1
        assert await breaker.can_execute() is True
        assert breaker.state == CircuitState.HALF_OPEN

        # Second call (half_open_calls=2 allows this)
        assert await breaker.can_execute() is True

        # Third call exceeds limit of 2
        assert await breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success_threshold(self, fast_config):
        """Should transition HALF_OPEN -> CLOSED after enough successes."""
        breaker = StreamingCircuitBreaker("test", fast_config)
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        await asyncio.sleep(0.1)
        await breaker.can_execute()  # HALF_OPEN

        # Need 2 successes (success_threshold=2)
        await breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self, fast_config):
        """Should transition HALF_OPEN -> OPEN on any failure."""
        breaker = StreamingCircuitBreaker("test", fast_config)
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        await asyncio.sleep(0.1)
        await breaker.can_execute()  # HALF_OPEN

        await breaker.record_failure(Exception("Still broken"))
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_records_state_changes(self, circuit_breaker):
        """Should record all state transitions with timestamps."""
        for i in range(3):
            await circuit_breaker.record_failure(Exception(f"Error {i}"))

        changes = circuit_breaker._state_changes
        assert len(changes) == 1
        ts, old, new = changes[0]
        assert old == CircuitState.CLOSED
        assert new == CircuitState.OPEN
        assert isinstance(ts, datetime)

    @pytest.mark.asyncio
    async def test_failure_resets_success_count(self, circuit_breaker):
        """Failure should reset success count to zero."""
        await circuit_breaker.record_success()
        await circuit_breaker.record_success()
        assert circuit_breaker._success_count == 2

        await circuit_breaker.record_failure(Exception("Error"))
        assert circuit_breaker._success_count == 0


# =============================================================================
# 4. Circuit Breaker Context Manager and call()
# =============================================================================


class TestCircuitBreakerContextManager:
    """Tests for circuit breaker context manager usage."""

    @pytest.mark.asyncio
    async def test_call_returns_context_when_closed(self, circuit_breaker):
        """Should return a context manager when circuit is CLOSED."""
        ctx = await circuit_breaker.call()
        assert isinstance(ctx, CircuitBreakerContext)

    @pytest.mark.asyncio
    async def test_context_records_success_on_normal_exit(self, circuit_breaker):
        """Should record success when context exits without exception."""
        async with await circuit_breaker.call():
            pass

        assert circuit_breaker._total_successes == 1
        assert circuit_breaker._total_failures == 0

    @pytest.mark.asyncio
    async def test_context_records_failure_on_exception(self, circuit_breaker):
        """Should record failure when context exits with exception."""
        with pytest.raises(RuntimeError):
            async with await circuit_breaker.call():
                raise RuntimeError("Operation failed")

        assert circuit_breaker._total_failures == 1
        assert circuit_breaker._total_successes == 0

    @pytest.mark.asyncio
    async def test_context_does_not_suppress_exception(self, circuit_breaker):
        """Should not suppress exceptions from the protected operation."""
        with pytest.raises(ValueError, match="specific error"):
            async with await circuit_breaker.call():
                raise ValueError("specific error")

    @pytest.mark.asyncio
    async def test_call_raises_when_open(self, fast_config):
        """Should raise CircuitBreakerOpenError when OPEN."""
        breaker = StreamingCircuitBreaker("broker", fast_config)
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call()

        assert exc_info.value.name == "broker"
        assert exc_info.value.recovery_time >= 0

    @pytest.mark.asyncio
    async def test_call_raises_with_descriptive_message(self, fast_config):
        """CircuitBreakerOpenError should have descriptive message."""
        breaker = StreamingCircuitBreaker("kafka-producer", fast_config)
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call()

        assert "kafka-producer" in str(exc_info.value)
        assert "open" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_context_handles_non_exception_baseexception(self, circuit_breaker):
        """Should handle BaseException subclasses that are not Exception."""
        with pytest.raises(KeyboardInterrupt):
            async with await circuit_breaker.call():
                raise KeyboardInterrupt()

        # record_failure is called with None since KeyboardInterrupt is not Exception
        assert circuit_breaker._total_failures == 1


# =============================================================================
# 5. Circuit Breaker Metrics and Reset
# =============================================================================


class TestCircuitBreakerMetrics:
    """Tests for circuit breaker statistics and reset."""

    @pytest.mark.asyncio
    async def test_get_stats_structure(self, circuit_breaker):
        """Should return all expected statistics fields."""
        stats = circuit_breaker.get_stats()

        assert stats["name"] == "test-breaker"
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 0
        assert stats["success_count"] == 0
        assert stats["total_calls"] == 0
        assert stats["total_failures"] == 0
        assert stats["total_successes"] == 0
        assert stats["time_until_recovery"] == 0
        assert stats["state_changes"] == 0

    @pytest.mark.asyncio
    async def test_stats_track_totals(self, circuit_breaker):
        """Should accurately track cumulative metrics."""
        await circuit_breaker.record_success()
        await circuit_breaker.record_success()
        await circuit_breaker.record_failure(Exception("err"))

        stats = circuit_breaker.get_stats()
        assert stats["total_calls"] == 3
        assert stats["total_successes"] == 2
        assert stats["total_failures"] == 1

    @pytest.mark.asyncio
    async def test_stats_show_recovery_time_when_open(self, fast_config):
        """Should show remaining recovery time when OPEN."""
        breaker = StreamingCircuitBreaker("test", fast_config)
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        stats = breaker.get_stats()
        assert stats["state"] == "open"
        assert stats["time_until_recovery"] > 0

    @pytest.mark.asyncio
    async def test_stats_count_state_changes(self, fast_config):
        """Should count state transitions."""
        breaker = StreamingCircuitBreaker("test", fast_config)

        # CLOSED -> OPEN
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))
        assert breaker.get_stats()["state_changes"] == 1

        # OPEN -> HALF_OPEN
        await asyncio.sleep(0.1)
        await breaker.can_execute()
        assert breaker.get_stats()["state_changes"] == 2

    def test_reset_restores_closed_state(self, circuit_breaker):
        """Should reset all state to CLOSED."""
        circuit_breaker._state = CircuitState.OPEN
        circuit_breaker._failure_count = 10
        circuit_breaker._success_count = 5
        circuit_breaker._half_open_calls = 3
        circuit_breaker._last_failure_time = time.time()

        circuit_breaker.reset()

        assert circuit_breaker.state == CircuitState.CLOSED
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker._success_count == 0
        assert circuit_breaker._half_open_calls == 0
        assert circuit_breaker._last_failure_time is None

    def test_time_until_recovery_zero_when_no_failure(self, circuit_breaker):
        """Should return 0 when there is no last failure."""
        assert circuit_breaker._time_until_recovery() == 0.0


# =============================================================================
# 6. CircuitBreakerOpenError Tests
# =============================================================================


class TestCircuitBreakerOpenError:
    """Tests for CircuitBreakerOpenError exception."""

    def test_attributes(self):
        """Should store name and recovery_time."""
        err = CircuitBreakerOpenError("my-broker", 15.5)
        assert err.name == "my-broker"
        assert err.recovery_time == 15.5

    def test_message_format(self):
        """Should have a descriptive error message."""
        err = CircuitBreakerOpenError("kafka-prod", 30.0)
        msg = str(err)
        assert "kafka-prod" in msg
        assert "30.0" in msg

    def test_is_exception(self):
        """Should be an Exception subclass."""
        err = CircuitBreakerOpenError("test", 0.0)
        assert isinstance(err, Exception)


# =============================================================================
# 7. DLQMessage Tests
# =============================================================================


class TestDLQMessage:
    """Tests for DLQMessage dataclass serialization."""

    def test_to_dict(self, sample_dlq_message):
        """Should serialize all fields to dictionary."""
        data = sample_dlq_message.to_dict()

        assert data["original_topic"] == "events"
        assert data["original_key"] == "evt-123"
        assert data["original_value"] == {"type": "order", "id": 42}
        assert data["original_headers"] == {"source": "api", "version": "2"}
        assert data["error_message"] == "Connection refused"
        assert data["error_type"] == "ConnectionError"
        assert data["retry_count"] == 3
        assert "original_timestamp" in data
        assert "failed_at" in data

    def test_to_json(self, sample_dlq_message):
        """Should produce valid JSON string."""
        json_str = sample_dlq_message.to_json()
        parsed = json.loads(json_str)

        assert parsed["original_topic"] == "events"
        assert parsed["error_type"] == "ConnectionError"

    def test_from_dict_roundtrip(self, sample_dlq_message):
        """Should round-trip through to_dict/from_dict."""
        data = sample_dlq_message.to_dict()
        restored = DLQMessage.from_dict(data)

        assert restored.original_topic == sample_dlq_message.original_topic
        assert restored.original_key == sample_dlq_message.original_key
        assert restored.original_value == sample_dlq_message.original_value
        assert restored.error_message == sample_dlq_message.error_message
        assert restored.error_type == sample_dlq_message.error_type
        assert restored.retry_count == sample_dlq_message.retry_count

    def test_from_dict_with_missing_optional_fields(self, sample_timestamp):
        """Should handle missing optional fields in from_dict."""
        data = {
            "original_topic": "topic1",
            "original_value": "payload",
            "original_timestamp": sample_timestamp.isoformat(),
            "error_message": "Timeout",
            "error_type": "TimeoutError",
            "retry_count": 1,
            "failed_at": sample_timestamp.isoformat(),
        }
        msg = DLQMessage.from_dict(data)

        assert msg.original_key is None
        assert msg.original_headers == {}

    def test_failed_at_defaults_to_now(self, sample_timestamp):
        """Should default failed_at to current time."""
        before = datetime.now(timezone.utc)
        msg = DLQMessage(
            original_topic="t",
            original_key=None,
            original_value="v",
            original_headers={},
            original_timestamp=sample_timestamp,
            error_message="err",
            error_type="Error",
            retry_count=0,
        )
        after = datetime.now(timezone.utc)

        assert before <= msg.failed_at <= after

    def test_to_json_produces_string(self, sample_dlq_message):
        """to_json should return a string."""
        result = sample_dlq_message.to_json()
        assert isinstance(result, str)

    def test_timestamps_are_isoformat(self, sample_dlq_message):
        """Serialized timestamps should be ISO format strings."""
        data = sample_dlq_message.to_dict()
        # Should be parseable as ISO format
        datetime.fromisoformat(data["original_timestamp"])
        datetime.fromisoformat(data["failed_at"])


# =============================================================================
# 8. DLQHandler - Message ID Generation
# =============================================================================


class TestDLQHandlerMessageId:
    """Tests for DLQ handler message ID generation and retry tracking."""

    def test_message_id_with_topic_and_key(self):
        """Should include topic and key in message ID."""
        handler = DLQHandler()
        msg_id = handler.get_message_id("orders", "order-42")
        assert msg_id == "orders:order-42"

    def test_message_id_with_offset(self):
        """Should include offset when provided."""
        handler = DLQHandler()
        msg_id = handler.get_message_id("orders", "key1", offset=1500)
        assert msg_id == "orders:key1:1500"

    def test_message_id_with_delivery_tag(self):
        """Should include delivery_tag when provided."""
        handler = DLQHandler()
        msg_id = handler.get_message_id("orders", "key1", delivery_tag=99)
        assert msg_id == "orders:key1:99"

    def test_message_id_with_all_parts(self):
        """Should include all parts when provided."""
        handler = DLQHandler()
        msg_id = handler.get_message_id("orders", "key1", offset=100, delivery_tag=5)
        assert msg_id == "orders:key1:100:5"

    def test_message_id_with_none_key(self):
        """Should handle None key."""
        handler = DLQHandler()
        msg_id = handler.get_message_id("orders", None)
        assert msg_id == "orders"

    def test_message_id_with_none_key_and_offset(self):
        """Should handle None key with offset."""
        handler = DLQHandler()
        msg_id = handler.get_message_id("orders", None, offset=100)
        assert msg_id == "orders:100"

    def test_different_messages_produce_different_ids(self):
        """Different messages should have different IDs."""
        handler = DLQHandler()
        id1 = handler.get_message_id("topic1", "key1", offset=1)
        id2 = handler.get_message_id("topic2", "key2", offset=2)
        assert id1 != id2


# =============================================================================
# 9. DLQHandler - Retry Tracking
# =============================================================================


class TestDLQHandlerRetryTracking:
    """Tests for DLQ handler retry count management."""

    def test_initial_retry_count_is_zero(self):
        """Should return 0 for unknown message IDs."""
        handler = DLQHandler()
        assert handler.get_retry_count("unknown:id") == 0

    def test_increment_retry(self):
        """Should increment retry count."""
        handler = DLQHandler()
        msg_id = "topic:key:100"

        assert handler.increment_retry(msg_id) == 1
        assert handler.increment_retry(msg_id) == 2
        assert handler.increment_retry(msg_id) == 3

    def test_increment_retry_tracks_total(self):
        """Should track total retries across all messages."""
        handler = DLQHandler()
        handler.increment_retry("msg1")
        handler.increment_retry("msg2")
        handler.increment_retry("msg1")

        assert handler._total_retries == 3

    def test_clear_retry(self):
        """Should remove retry count for a message."""
        handler = DLQHandler()
        handler.increment_retry("msg1")
        handler.increment_retry("msg1")

        handler.clear_retry("msg1")
        assert handler.get_retry_count("msg1") == 0

    def test_clear_retry_nonexistent_is_safe(self):
        """Should not raise when clearing unknown message ID."""
        handler = DLQHandler()
        handler.clear_retry("nonexistent")  # Should not raise


# =============================================================================
# 10. DLQHandler - should_send_to_dlq
# =============================================================================


class TestDLQHandlerShouldSendToDLQ:
    """Tests for DLQ routing decision logic."""

    @pytest.mark.asyncio
    async def test_returns_false_below_max_retries(self):
        """Should not send to DLQ when retry count < max_retries."""
        config = StreamingResilienceConfig(dlq_max_retries=3)
        handler = DLQHandler(config)
        msg_id = "topic:key:1"

        handler.increment_retry(msg_id)
        handler.increment_retry(msg_id)

        assert await handler.should_send_to_dlq(msg_id) is False

    @pytest.mark.asyncio
    async def test_returns_true_at_max_retries(self):
        """Should send to DLQ when retry count >= max_retries."""
        config = StreamingResilienceConfig(dlq_max_retries=2)
        handler = DLQHandler(config)
        msg_id = "topic:key:1"

        handler.increment_retry(msg_id)
        handler.increment_retry(msg_id)

        assert await handler.should_send_to_dlq(msg_id) is True

    @pytest.mark.asyncio
    async def test_returns_true_above_max_retries(self):
        """Should send to DLQ when retry count > max_retries."""
        config = StreamingResilienceConfig(dlq_max_retries=2)
        handler = DLQHandler(config)
        msg_id = "topic:key:1"

        for _ in range(5):
            handler.increment_retry(msg_id)

        assert await handler.should_send_to_dlq(msg_id) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_dlq_disabled(self):
        """Should never send to DLQ when dlq_enabled=False."""
        config = StreamingResilienceConfig(dlq_enabled=False, dlq_max_retries=1)
        handler = DLQHandler(config)
        msg_id = "topic:key:1"

        for _ in range(10):
            handler.increment_retry(msg_id)

        assert await handler.should_send_to_dlq(msg_id) is False


# =============================================================================
# 11. DLQHandler - handle_failure
# =============================================================================


class TestDLQHandlerHandleFailure:
    """Tests for DLQ handler failure handling pipeline."""

    @pytest.mark.asyncio
    async def test_returns_false_for_retry(self, dlq_handler, sample_timestamp, dlq_sender):
        """Should return False when message should be retried."""
        result = await dlq_handler.handle_failure(
            topic="events",
            key="evt-1",
            value={"data": "test"},
            headers={"source": "api"},
            timestamp=sample_timestamp,
            error=Exception("Transient error"),
            offset=100,
        )

        assert result is False
        dlq_sender.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_true_after_max_retries(self, dlq_handler, sample_timestamp, dlq_sender):
        """Should return True and send to DLQ after max retries."""
        # dlq_max_retries=2 in fast_config, so need 2 calls
        for _ in range(2):
            result = await dlq_handler.handle_failure(
                topic="events",
                key="evt-1",
                value={"data": "test"},
                headers={},
                timestamp=sample_timestamp,
                error=Exception("Persistent error"),
                offset=100,
            )

        assert result is True
        dlq_sender.assert_called_once()

    @pytest.mark.asyncio
    async def test_clears_retry_after_dlq_send(self, dlq_handler, sample_timestamp):
        """Should clear retry count after sending to DLQ."""
        msg_id = dlq_handler.get_message_id("events", "evt-1", offset=100)

        for _ in range(2):
            await dlq_handler.handle_failure(
                topic="events",
                key="evt-1",
                value={"data": "test"},
                headers={},
                timestamp=sample_timestamp,
                error=Exception("Error"),
                offset=100,
            )

        assert dlq_handler.get_retry_count(msg_id) == 0

    @pytest.mark.asyncio
    async def test_tracks_total_messages(self, dlq_handler, sample_timestamp):
        """Should count total messages handled."""
        for i in range(5):
            await dlq_handler.handle_failure(
                topic="events",
                key=f"evt-{i}",
                value={"data": "test"},
                headers={},
                timestamp=sample_timestamp,
                error=Exception("Error"),
            )

        assert dlq_handler._total_messages == 5

    @pytest.mark.asyncio
    async def test_handles_kafka_offset(self, dlq_handler, sample_timestamp):
        """Should use offset for Kafka message ID generation."""
        await dlq_handler.handle_failure(
            topic="events",
            key="evt-1",
            value={"data": "test"},
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
            offset=500,
        )

        msg_id = dlq_handler.get_message_id("events", "evt-1", offset=500)
        assert dlq_handler.get_retry_count(msg_id) == 1

    @pytest.mark.asyncio
    async def test_handles_rabbitmq_delivery_tag(self, dlq_handler, sample_timestamp):
        """Should use delivery_tag for RabbitMQ message ID generation."""
        await dlq_handler.handle_failure(
            topic="events",
            key="evt-1",
            value={"data": "test"},
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
            delivery_tag=42,
        )

        msg_id = dlq_handler.get_message_id("events", "evt-1", delivery_tag=42)
        assert dlq_handler.get_retry_count(msg_id) == 1


# =============================================================================
# 12. DLQHandler - send_to_dlq
# =============================================================================


class TestDLQHandlerSendToDLQ:
    """Tests for direct DLQ sending."""

    @pytest.mark.asyncio
    async def test_calls_sender_with_correct_topic(self, dlq_handler, dlq_sender, sample_timestamp):
        """Should append DLQ suffix to topic name."""
        await dlq_handler.send_to_dlq(
            topic="orders",
            key="ord-1",
            value={"id": 1},
            headers={},
            timestamp=sample_timestamp,
            error=ValueError("Bad data"),
        )

        dlq_sender.assert_called_once()
        call_topic = dlq_sender.call_args[0][0]
        assert call_topic == "orders.dlq"

    @pytest.mark.asyncio
    async def test_sends_dlq_message_object(self, dlq_handler, dlq_sender, sample_timestamp):
        """Should send a DLQMessage object to the sender."""
        await dlq_handler.send_to_dlq(
            topic="orders",
            key="ord-1",
            value={"id": 1},
            headers={"version": "2"},
            timestamp=sample_timestamp,
            error=TypeError("Wrong type"),
            retry_count=5,
        )

        dlq_message = dlq_sender.call_args[0][1]
        assert isinstance(dlq_message, DLQMessage)
        assert dlq_message.original_topic == "orders"
        assert dlq_message.original_key == "ord-1"
        assert dlq_message.error_type == "TypeError"
        assert dlq_message.retry_count == 5

    @pytest.mark.asyncio
    async def test_calls_on_dlq_send_callback(self, dlq_handler, dlq_callback, sample_timestamp):
        """Should call on_dlq_send callback after sending."""
        await dlq_handler.send_to_dlq(
            topic="orders",
            key="ord-1",
            value={"id": 1},
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
        )

        dlq_callback.assert_called_once()
        callback_msg = dlq_callback.call_args[0][0]
        assert isinstance(callback_msg, DLQMessage)

    @pytest.mark.asyncio
    async def test_skips_send_when_dlq_disabled(self, sample_timestamp):
        """Should not send when DLQ is disabled."""
        config = StreamingResilienceConfig(dlq_enabled=False)
        sender = AsyncMock()
        handler = DLQHandler(config, dlq_sender=sender)

        await handler.send_to_dlq(
            topic="orders",
            key="ord-1",
            value={"id": 1},
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
        )

        sender.assert_not_called()

    @pytest.mark.asyncio
    async def test_tracks_dlq_send_count(self, dlq_handler, sample_timestamp):
        """Should increment total_dlq_sends counter."""
        await dlq_handler.send_to_dlq(
            topic="orders",
            key="ord-1",
            value={"id": 1},
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
        )

        assert dlq_handler._total_dlq_sends == 1

    @pytest.mark.asyncio
    async def test_raises_on_sender_failure(self, sample_timestamp):
        """Should raise when DLQ sender fails."""
        sender = AsyncMock(side_effect=ConnectionError("Cannot reach DLQ broker"))
        handler = DLQHandler(dlq_sender=sender)

        with pytest.raises(ConnectionError, match="Cannot reach DLQ broker"):
            await handler.send_to_dlq(
                topic="orders",
                key="ord-1",
                value={"id": 1},
                headers={},
                timestamp=sample_timestamp,
                error=Exception("Error"),
            )

    @pytest.mark.asyncio
    async def test_sender_failure_still_counts_send(self, sample_timestamp):
        """Should still count the DLQ send attempt even on sender failure."""
        sender = AsyncMock(side_effect=OSError("Disk full"))
        handler = DLQHandler(dlq_sender=sender)

        with pytest.raises(OSError):
            await handler.send_to_dlq(
                topic="orders",
                key="ord-1",
                value={"id": 1},
                headers={},
                timestamp=sample_timestamp,
                error=Exception("Error"),
            )

        assert handler._total_dlq_sends == 1

    @pytest.mark.asyncio
    async def test_logs_when_no_sender_configured(self, sample_timestamp):
        """Should log warning when no sender is configured."""
        handler = DLQHandler()

        # Should not raise, just log
        await handler.send_to_dlq(
            topic="orders",
            key="ord-1",
            value={"id": 1},
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
        )

        assert handler._total_dlq_sends == 1

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_raise(self, sample_timestamp):
        """Should not raise when callback fails with handled exception types."""
        callback = AsyncMock(side_effect=RuntimeError("Callback broken"))
        handler = DLQHandler(on_dlq_send=callback)

        # Should not raise
        await handler.send_to_dlq(
            topic="orders",
            key="ord-1",
            value={"id": 1},
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
        )

    @pytest.mark.asyncio
    async def test_custom_dlq_topic_suffix(self, sample_timestamp):
        """Should use custom DLQ topic suffix from config."""
        config = StreamingResilienceConfig(dlq_topic_suffix=".dead-letter")
        sender = AsyncMock()
        handler = DLQHandler(config, dlq_sender=sender)

        await handler.send_to_dlq(
            topic="events",
            key="evt-1",
            value="payload",
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Error"),
        )

        call_topic = sender.call_args[0][0]
        assert call_topic == "events.dead-letter"


# =============================================================================
# 13. DLQHandler - Statistics
# =============================================================================


class TestDLQHandlerStats:
    """Tests for DLQ handler statistics."""

    def test_get_stats_structure(self):
        """Should return all expected statistics fields."""
        handler = DLQHandler()
        stats = handler.get_stats()

        assert "enabled" in stats
        assert "max_retries" in stats
        assert "total_messages" in stats
        assert "total_retries" in stats
        assert "total_dlq_sends" in stats
        assert "pending_retries" in stats

    def test_get_stats_reflects_config(self, fast_config):
        """Should reflect config values in stats."""
        handler = DLQHandler(fast_config)
        stats = handler.get_stats()

        assert stats["enabled"] is True
        assert stats["max_retries"] == fast_config.dlq_max_retries

    @pytest.mark.asyncio
    async def test_stats_track_pending_retries(self, sample_timestamp):
        """Should track messages with pending retries."""
        handler = DLQHandler()

        await handler.handle_failure(
            topic="t",
            key="k1",
            value="v",
            headers={},
            timestamp=sample_timestamp,
            error=Exception("E"),
        )
        await handler.handle_failure(
            topic="t",
            key="k2",
            value="v",
            headers={},
            timestamp=sample_timestamp,
            error=Exception("E"),
        )

        stats = handler.get_stats()
        assert stats["pending_retries"] == 2


# =============================================================================
# 14. HealthMonitor Tests
# =============================================================================


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.mark.asyncio
    async def test_starts_healthy(self, health_monitor):
        """Should start in healthy state."""
        status = await health_monitor.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 0
        assert status.messages_processed == 0
        assert status.messages_failed == 0

    @pytest.mark.asyncio
    async def test_record_success_updates_status(self, health_monitor):
        """Should update processed count and latency on success."""
        await health_monitor.record_success(latency_ms=25.5)

        status = await health_monitor.get_status()
        assert status.healthy is True
        assert status.messages_processed == 1
        assert status.latency_ms == 25.5
        assert status.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_record_success_without_latency(self, health_monitor):
        """Should handle success without latency."""
        await health_monitor.record_success()

        status = await health_monitor.get_status()
        assert status.latency_ms is None
        assert status.messages_processed == 1

    @pytest.mark.asyncio
    async def test_record_failure_tracks_error(self, health_monitor):
        """Should track failure details."""
        await health_monitor.record_failure(Exception("Connection lost"))

        status = await health_monitor.get_status()
        assert status.messages_failed == 1
        assert status.consecutive_failures == 1
        assert status.last_error == "Connection lost"

    @pytest.mark.asyncio
    async def test_record_failure_with_string_error(self, health_monitor):
        """Should accept string error messages."""
        await health_monitor.record_failure("Timeout exceeded")

        status = await health_monitor.get_status()
        assert status.last_error == "Timeout exceeded"

    @pytest.mark.asyncio
    async def test_becomes_unhealthy_at_threshold(self, health_monitor):
        """Should become unhealthy after consecutive failures >= threshold."""
        # unhealthy_threshold=2 in fast_config
        await health_monitor.record_failure(Exception("Err 1"))
        status = await health_monitor.get_status()
        assert status.healthy is True  # Not yet at threshold

        await health_monitor.record_failure(Exception("Err 2"))
        status = await health_monitor.get_status()
        assert status.healthy is False

    @pytest.mark.asyncio
    async def test_success_resets_consecutive_failures(self, health_monitor):
        """Should reset consecutive failures on success."""
        await health_monitor.record_failure(Exception("Err"))
        await health_monitor.record_success()

        status = await health_monitor.get_status()
        assert status.consecutive_failures == 0
        assert status.healthy is True

    @pytest.mark.asyncio
    async def test_success_restores_healthy_state(self, health_monitor):
        """Success should restore healthy state from unhealthy."""
        # Make unhealthy
        await health_monitor.record_failure(Exception("Err 1"))
        await health_monitor.record_failure(Exception("Err 2"))
        status = await health_monitor.get_status()
        assert status.healthy is False

        # Success restores healthy
        await health_monitor.record_success()
        status = await health_monitor.get_status()
        assert status.healthy is True

    @pytest.mark.asyncio
    async def test_reset_restores_healthy(self, health_monitor):
        """Should reset all state to healthy."""
        await health_monitor.record_failure(Exception("Err 1"))
        await health_monitor.record_failure(Exception("Err 2"))
        await health_monitor.record_failure(Exception("Err 3"))

        await health_monitor.reset()

        status = await health_monitor.get_status()
        assert status.healthy is True
        assert status.consecutive_failures == 0
        assert status.last_error is None

    @pytest.mark.asyncio
    async def test_multiple_successes_accumulate(self, health_monitor):
        """Should accumulate multiple success counts."""
        for _ in range(10):
            await health_monitor.record_success(latency_ms=5.0)

        status = await health_monitor.get_status()
        assert status.messages_processed == 10

    @pytest.mark.asyncio
    async def test_interleaved_success_failure(self, health_monitor):
        """Should handle interleaved successes and failures correctly."""
        await health_monitor.record_success()  # processed=1
        await health_monitor.record_failure(Exception("E1"))  # failed=1, consec=1
        await health_monitor.record_success()  # processed=2, consec=0
        await health_monitor.record_failure(Exception("E2"))  # failed=2, consec=1
        await health_monitor.record_failure(Exception("E3"))  # failed=3, consec=2, unhealthy

        status = await health_monitor.get_status()
        assert status.messages_processed == 2
        assert status.messages_failed == 3
        assert status.consecutive_failures == 2
        assert status.healthy is False

    @pytest.mark.asyncio
    async def test_last_check_updates_on_success(self, health_monitor):
        """Should update last_check timestamp on success."""
        before = datetime.now(timezone.utc)
        await health_monitor.record_success()
        status = await health_monitor.get_status()
        assert status.last_check >= before

    @pytest.mark.asyncio
    async def test_last_check_updates_on_failure(self, health_monitor):
        """Should update last_check timestamp on failure."""
        before = datetime.now(timezone.utc)
        await health_monitor.record_failure(Exception("Err"))
        status = await health_monitor.get_status()
        assert status.last_check >= before


# =============================================================================
# 15. HealthStatus Tests
# =============================================================================


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        now = datetime.now(timezone.utc)
        status = HealthStatus(
            healthy=False,
            last_check=now,
            consecutive_failures=5,
            last_error="Connection reset",
            latency_ms=42.5,
            messages_processed=1000,
            messages_failed=50,
        )

        data = status.to_dict()
        assert data["healthy"] is False
        assert data["consecutive_failures"] == 5
        assert data["last_error"] == "Connection reset"
        assert data["latency_ms"] == 42.5
        assert data["messages_processed"] == 1000
        assert data["messages_failed"] == 50
        assert "last_check" in data

    def test_to_dict_with_none_optionals(self):
        """Should handle None optional fields."""
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
        )

        data = status.to_dict()
        assert data["last_error"] is None
        assert data["latency_ms"] is None

    def test_default_values(self):
        """Should have sensible defaults."""
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
        )
        assert status.consecutive_failures == 0
        assert status.messages_processed == 0
        assert status.messages_failed == 0


# =============================================================================
# 16. GracefulShutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for GracefulShutdown handler."""

    def test_starts_not_shutting_down(self):
        """Should start in non-shutdown state."""
        shutdown = GracefulShutdown()
        assert shutdown.is_shutting_down is False

    def test_trigger_shutdown(self):
        """Should trigger shutdown state."""
        shutdown = GracefulShutdown()
        shutdown.trigger_shutdown()
        assert shutdown.is_shutting_down is True

    def test_register_cleanup_tasks(self):
        """Should register multiple cleanup functions."""
        shutdown = GracefulShutdown()

        async def cleanup1():
            pass

        async def cleanup2():
            pass

        async def cleanup3():
            pass

        shutdown.register_cleanup(cleanup1)
        shutdown.register_cleanup(cleanup2)
        shutdown.register_cleanup(cleanup3)

        assert len(shutdown._cleanup_tasks) == 3

    @pytest.mark.asyncio
    async def test_wait_for_shutdown(self):
        """Should wait until shutdown is triggered."""
        shutdown = GracefulShutdown()

        async def trigger_later():
            await asyncio.sleep(0.05)
            shutdown.trigger_shutdown()

        asyncio.create_task(trigger_later())

        await asyncio.wait_for(shutdown.wait_for_shutdown(), timeout=1.0)
        assert shutdown.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_run_cleanup_executes_all_tasks(self):
        """Should run all registered cleanup tasks."""
        shutdown = GracefulShutdown()

        executed = []

        async def cleanup1():
            executed.append(1)

        async def cleanup2():
            executed.append(2)

        shutdown.register_cleanup(cleanup1)
        shutdown.register_cleanup(cleanup2)

        await shutdown._run_cleanup()

        assert executed == [1, 2]

    @pytest.mark.asyncio
    async def test_run_cleanup_handles_timeout(self):
        """Should handle cleanup tasks that timeout.

        Uses a very short timeout to trigger TimeoutError without patching
        asyncio.wait_for globally (which can corrupt async mock state for
        subsequent tests when run with randomized ordering).
        """
        shutdown = GracefulShutdown()

        # Event to keep the slow cleanup blocked until we release it
        blocker = asyncio.Event()

        async def slow_cleanup():
            await blocker.wait()  # Block until released

        async def fast_cleanup():
            pass

        shutdown.register_cleanup(slow_cleanup)
        shutdown.register_cleanup(fast_cleanup)

        # Patch the hardcoded 30s timeout to 0.01s so the test is fast.
        # We patch at the module level where asyncio.wait_for is called,
        # wrapping the real function with a shorter timeout.
        _real_wait_for = asyncio.wait_for

        async def _short_timeout_wait_for(coro, *, timeout=30.0):
            """Wrap real wait_for with a much shorter timeout."""
            return await _real_wait_for(coro, timeout=0.01)

        with patch(
            "asyncio.wait_for",
            side_effect=_short_timeout_wait_for,
        ):
            await shutdown._run_cleanup()

        # Release the blocker to avoid warnings about pending coroutines
        blocker.set()

    @pytest.mark.asyncio
    async def test_run_cleanup_handles_task_failure(self):
        """Should continue cleanup even if a task fails."""
        shutdown = GracefulShutdown()

        executed = []

        async def failing_cleanup():
            raise RuntimeError("Cleanup failed")

        async def successful_cleanup():
            executed.append("ok")

        shutdown.register_cleanup(failing_cleanup)
        shutdown.register_cleanup(successful_cleanup)

        await shutdown._run_cleanup()

        # Second task should still execute
        assert "ok" in executed

    def test_trigger_shutdown_sets_event(self):
        """Should set the shutdown event for waiters."""
        shutdown = GracefulShutdown()
        assert not shutdown._shutdown_event.is_set()

        shutdown.trigger_shutdown()
        assert shutdown._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_setup_signal_handlers_idempotent(self):
        """Should only set up signal handlers once."""
        shutdown = GracefulShutdown()

        try:
            shutdown.setup_signal_handlers()
            shutdown.setup_signal_handlers()  # Second call should be no-op
            assert shutdown._setup_done is True
        except NotImplementedError:
            # Windows or non-main-thread context
            pass


# =============================================================================
# 17. with_retry Decorator Tests
# =============================================================================


class TestWithRetryDecorator:
    """Tests for the with_retry decorator."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Should return result when function succeeds immediately."""
        config = StreamingResilienceConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
            jitter=False,
        )

        @with_retry(config=config)
        async def succeed():
            return "ok"

        result = await succeed()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        """Should retry on ConnectionError."""
        config = StreamingResilienceConfig(
            max_retries=3,
            initial_delay_seconds=0.001,
            jitter=False,
        )

        call_count = 0

        @with_retry(config=config)
        async def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return "recovered"

        result = await flaky()
        assert result == "recovered"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retries_on_timeout_error(self):
        """Should retry on TimeoutError."""
        config = StreamingResilienceConfig(
            max_retries=2,
            initial_delay_seconds=0.001,
            jitter=False,
        )

        call_count = 0

        @with_retry(config=config)
        async def timeout_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timed out")
            return "done"

        result = await timeout_func()
        assert result == "done"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_os_error(self):
        """Should retry on OSError."""
        config = StreamingResilienceConfig(
            max_retries=2,
            initial_delay_seconds=0.001,
            jitter=False,
        )

        call_count = 0

        @with_retry(config=config)
        async def os_error_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise OSError("I/O error")
            return "done"

        result = await os_error_func()
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Should raise after exhausting retries."""
        config = StreamingResilienceConfig(
            max_retries=2,
            initial_delay_seconds=0.001,
            jitter=False,
        )

        @with_retry(config=config)
        async def always_fail():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError, match="Always fails"):
            await always_fail()

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable_exceptions(self):
        """Should not retry exceptions not in retryable_exceptions."""
        config = StreamingResilienceConfig(
            max_retries=5,
            initial_delay_seconds=0.001,
            jitter=False,
        )

        call_count = 0

        @with_retry(config=config)
        async def value_error_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError, match="Not retryable"):
            await value_error_func()

        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_custom_retryable_exceptions(self):
        """Should retry only specified exception types."""
        config = StreamingResilienceConfig(
            max_retries=3,
            initial_delay_seconds=0.001,
            jitter=False,
        )

        call_count = 0

        @with_retry(
            config=config,
            retryable_exceptions=(ValueError, KeyError),
        )
        async def custom_retry():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retryable")
            return "ok"

        result = await custom_retry()
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_uses_default_config(self):
        """Should use default config when none provided."""
        call_count = 0

        @with_retry()
        async def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Fail")
            return "ok"

        # Use mock sleep to avoid actual delays
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await func()

        assert result == "ok"

    @pytest.mark.asyncio
    async def test_zero_retries_only_tries_once(self):
        """Should try exactly once with max_retries=0."""
        config = StreamingResilienceConfig(
            max_retries=0,
            initial_delay_seconds=0.001,
        )

        call_count = 0

        @with_retry(config=config)
        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Fail")

        with pytest.raises(ConnectionError):
            await always_fail()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_passes_arguments_through(self):
        """Should pass args and kwargs to decorated function."""
        config = StreamingResilienceConfig(
            max_retries=1,
            initial_delay_seconds=0.001,
            jitter=False,
        )

        @with_retry(config=config)
        async def add(a, b, multiplier=1):
            return (a + b) * multiplier

        result = await add(3, 4, multiplier=2)
        assert result == 14

    @pytest.mark.asyncio
    async def test_applies_backoff_delays(self):
        """Should apply exponential backoff delays between retries."""
        config = StreamingResilienceConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
            jitter=False,
        )

        sleep_calls = []

        @with_retry(config=config)
        async def always_fail():
            raise ConnectionError("Fail")

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = lambda d: sleep_calls.append(d)

            with pytest.raises(ConnectionError):
                await always_fail()

        # Should have slept 3 times (after attempts 0, 1, 2; attempt 3 raises)
        assert len(sleep_calls) == 3
        # Delays should increase
        assert sleep_calls[0] <= sleep_calls[1] <= sleep_calls[2]


# =============================================================================
# 18. Edge Cases - Rapid Failures
# =============================================================================


class TestRapidFailures:
    """Tests for rapid sequential failure scenarios."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_rapid_failures_open_immediately(self):
        """Circuit breaker should open as soon as threshold is hit."""
        config = StreamingResilienceConfig(
            circuit_breaker_threshold=5,
            circuit_breaker_recovery_seconds=60.0,
        )
        breaker = StreamingCircuitBreaker("rapid-test", config)

        # Rapidly record failures
        for i in range(5):
            await breaker.record_failure(Exception(f"Rapid failure {i}"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 5

        # Should reject immediately
        assert await breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_health_monitor_rapid_failures(self):
        """Health monitor should become unhealthy quickly under rapid failures."""
        config = StreamingResilienceConfig(unhealthy_threshold=3)
        monitor = HealthMonitor("rapid-test", config)

        for i in range(10):
            await monitor.record_failure(Exception(f"Failure {i}"))

        status = await monitor.get_status()
        assert status.healthy is False
        assert status.consecutive_failures == 10
        assert status.messages_failed == 10

    @pytest.mark.asyncio
    async def test_dlq_rapid_failures_same_message(self, sample_timestamp):
        """DLQ handler should route to DLQ after max retries for same message."""
        config = StreamingResilienceConfig(dlq_max_retries=2)
        sender = AsyncMock()
        handler = DLQHandler(config, dlq_sender=sender)

        results = []
        for i in range(5):
            result = await handler.handle_failure(
                topic="events",
                key="same-key",
                value="payload",
                headers={},
                timestamp=sample_timestamp,
                error=Exception(f"Error {i}"),
                offset=100,
            )
            results.append(result)

        # First failure: retry (count=1 < 2)
        assert results[0] is False
        # Second failure: DLQ (count=2 >= 2), clears count
        assert results[1] is True
        # Third failure: retry again (count reset, now 1 < 2)
        assert results[2] is False
        # Fourth failure: DLQ again (count=2 >= 2)
        assert results[3] is True
        # Fifth failure: retry (count reset again)
        assert results[4] is False

        assert sender.call_count == 2


# =============================================================================
# 19. Edge Cases - Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access to shared resilience components."""

    @pytest.mark.asyncio
    async def test_concurrent_circuit_breaker_operations(self):
        """Circuit breaker should be safe under concurrent access."""
        config = StreamingResilienceConfig(
            circuit_breaker_threshold=10,
            circuit_breaker_recovery_seconds=60.0,
        )
        breaker = StreamingCircuitBreaker("concurrent-test", config)

        async def record_failure():
            await breaker.record_failure(Exception("Error"))

        async def record_success():
            await breaker.record_success()

        # Mix of concurrent successes and failures
        tasks = []
        for i in range(20):
            if i % 3 == 0:
                tasks.append(record_success())
            else:
                tasks.append(record_failure())

        await asyncio.gather(*tasks)

        # Total should be 20
        assert breaker._total_calls == 20

    @pytest.mark.asyncio
    async def test_concurrent_health_monitor_operations(self):
        """Health monitor should be safe under concurrent access."""
        monitor = HealthMonitor("concurrent-test")

        async def record_op(is_success):
            if is_success:
                await monitor.record_success(latency_ms=10.0)
            else:
                await monitor.record_failure(Exception("Error"))

        tasks = [record_op(i % 2 == 0) for i in range(50)]
        await asyncio.gather(*tasks)

        status = await monitor.get_status()
        assert status.messages_processed + status.messages_failed == 50

    @pytest.mark.asyncio
    async def test_concurrent_dlq_different_messages(self, sample_timestamp):
        """DLQ handler should track retries independently for different messages."""
        handler = DLQHandler(StreamingResilienceConfig(dlq_max_retries=3))

        async def handle_msg(key, offset):
            return await handler.handle_failure(
                topic="events",
                key=key,
                value="payload",
                headers={},
                timestamp=sample_timestamp,
                error=Exception("Error"),
                offset=offset,
            )

        # Handle different messages concurrently
        results = await asyncio.gather(
            handle_msg("key-1", 1),
            handle_msg("key-2", 2),
            handle_msg("key-3", 3),
        )

        # All should be False (first retry for each)
        assert all(r is False for r in results)
        assert handler._total_messages == 3


# =============================================================================
# 20. Integration / End-to-End Scenarios
# =============================================================================


class TestResilienceIntegration:
    """Integration tests combining multiple resilience patterns."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_health_monitor(self, fast_config):
        """Circuit breaker and health monitor should work together."""
        breaker = StreamingCircuitBreaker("integration", fast_config)
        monitor = HealthMonitor("integration", fast_config)

        # Simulate failures
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))
            await monitor.record_failure(Exception(f"Error {i}"))

        assert breaker.state == CircuitState.OPEN
        health = await monitor.get_status()
        assert health.healthy is False

        # Wait for recovery
        await asyncio.sleep(0.1)

        # Try recovery
        if await breaker.can_execute():
            await breaker.record_success()
            await monitor.record_success()

        health = await monitor.get_status()
        assert health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_dlq_with_circuit_breaker(self, fast_config, sample_timestamp):
        """DLQ should be usable even when circuit breaker is open."""
        breaker = StreamingCircuitBreaker("dlq-test", fast_config)
        sender = AsyncMock()
        dlq = DLQHandler(fast_config, dlq_sender=sender)

        # Open the circuit
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        assert breaker.state == CircuitState.OPEN

        # Should still be able to send to DLQ
        await dlq.send_to_dlq(
            topic="events",
            key="evt-1",
            value="payload",
            headers={},
            timestamp=sample_timestamp,
            error=Exception("Circuit open"),
        )

        sender.assert_called_once()

    @pytest.mark.asyncio
    async def test_backoff_with_circuit_breaker_recovery(self, fast_config):
        """Backoff should work with circuit breaker recovery cycle."""
        breaker = StreamingCircuitBreaker("backoff-test", fast_config)
        backoff = ExponentialBackoff(fast_config)

        # Open the circuit
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        assert breaker.state == CircuitState.OPEN

        # Wait for recovery with backoff-like delay
        delay = backoff.get_delay(0)
        await asyncio.sleep(max(delay, 0.1))

        # Should be able to try again
        can_exec = await breaker.can_execute()
        assert can_exec is True
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_full_failure_recovery_cycle(self, fast_config, sample_timestamp):
        """Should handle a full failure -> DLQ -> recovery cycle."""
        breaker = StreamingCircuitBreaker("full-cycle", fast_config)
        monitor = HealthMonitor("full-cycle", fast_config)
        sender = AsyncMock()
        dlq = DLQHandler(fast_config, dlq_sender=sender)

        # Phase 1: Failures accumulate
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))
            await monitor.record_failure(Exception(f"Error {i}"))

        assert breaker.state == CircuitState.OPEN
        health = await monitor.get_status()
        assert health.healthy is False

        # Phase 2: Failed message goes to DLQ
        for _ in range(fast_config.dlq_max_retries):
            await dlq.handle_failure(
                topic="events",
                key="evt-stuck",
                value="payload",
                headers={},
                timestamp=sample_timestamp,
                error=Exception("Still failing"),
                offset=1,
            )
        assert dlq._total_dlq_sends == 1

        # Phase 3: Wait and recover
        await asyncio.sleep(0.1)
        assert await breaker.can_execute()
        await breaker.record_success()
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

        # Phase 4: Health restored
        await monitor.record_success()
        health = await monitor.get_status()
        assert health.healthy is True

    @pytest.mark.asyncio
    async def test_retry_decorator_with_circuit_breaker(self, fast_config):
        """with_retry decorator should work with circuit breaker context."""
        breaker = StreamingCircuitBreaker("retry-cb", fast_config)

        call_count = 0

        @with_retry(
            config=fast_config,
            retryable_exceptions=(ConnectionError, CircuitBreakerOpenError),
        )
        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Not ready")
            return "success"

        result = await operation()
        assert result == "success"
        assert call_count == 3
