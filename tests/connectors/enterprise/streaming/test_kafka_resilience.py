"""
Tests for Kafka Connector Resilience Patterns.

Tests cover:
- Connection retry with exponential backoff
- Circuit breaker state transitions
- Dead letter queue handling
- Graceful shutdown
- Health monitoring
- Integration with KafkaConnector

These tests mock the aiokafka library to avoid requiring
an actual Kafka cluster.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.enterprise.streaming.resilience import (
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
)


# =============================================================================
# StreamingResilienceConfig Tests
# =============================================================================


class TestStreamingResilienceConfig:
    """Tests for StreamingResilienceConfig."""

    def test_default_config(self):
        """Should initialize with sensible defaults."""
        config = StreamingResilienceConfig()

        assert config.max_retries == 5
        assert config.initial_delay_seconds == 1.0
        assert config.max_delay_seconds == 60.0
        assert config.circuit_breaker_threshold == 5
        assert config.dlq_enabled is True
        assert config.dlq_max_retries == 3

    def test_custom_config(self):
        """Should accept custom values."""
        config = StreamingResilienceConfig(
            max_retries=10,
            initial_delay_seconds=0.5,
            circuit_breaker_threshold=3,
            dlq_enabled=False,
        )

        assert config.max_retries == 10
        assert config.initial_delay_seconds == 0.5
        assert config.circuit_breaker_threshold == 3
        assert config.dlq_enabled is False

    def test_validation_max_retries(self):
        """Should reject negative max_retries."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            StreamingResilienceConfig(max_retries=-1)

    def test_validation_initial_delay(self):
        """Should reject non-positive initial_delay_seconds."""
        with pytest.raises(ValueError, match="initial_delay_seconds must be positive"):
            StreamingResilienceConfig(initial_delay_seconds=0)

    def test_validation_max_delay(self):
        """Should reject max_delay < initial_delay."""
        with pytest.raises(ValueError, match="max_delay_seconds must be >= initial_delay"):
            StreamingResilienceConfig(
                initial_delay_seconds=10.0,
                max_delay_seconds=5.0,
            )


# =============================================================================
# ExponentialBackoff Tests
# =============================================================================


class TestExponentialBackoff:
    """Tests for ExponentialBackoff."""

    def test_first_delay(self):
        """Should return initial delay for first attempt."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=1.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        delay = backoff.get_delay(0)
        assert delay == 1.0

    def test_exponential_growth(self):
        """Should grow exponentially with attempts."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=1.0,
            exponential_base=2.0,
            max_delay_seconds=60.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        assert backoff.get_delay(0) == 1.0
        assert backoff.get_delay(1) == 2.0
        assert backoff.get_delay(2) == 4.0
        assert backoff.get_delay(3) == 8.0

    def test_max_delay_cap(self):
        """Should cap delay at max_delay_seconds."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=1.0,
            max_delay_seconds=5.0,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        delay = backoff.get_delay(10)  # Would be 1024 without cap
        assert delay == 5.0

    def test_jitter_adds_randomness(self):
        """Should add jitter when enabled."""
        config = StreamingResilienceConfig(
            initial_delay_seconds=1.0,
            jitter=True,
        )
        backoff = ExponentialBackoff(config)

        delays = [backoff.get_delay(0) for _ in range(10)]
        # With jitter, delays should vary
        unique_delays = set(delays)
        assert len(unique_delays) > 1

    def test_reset(self):
        """Should reset state."""
        config = StreamingResilienceConfig(initial_delay_seconds=1.0, jitter=False)
        backoff = ExponentialBackoff(config)

        backoff.get_delay(5)
        backoff.reset()

        # After reset, should behave like new
        assert backoff._attempt == 0

    @pytest.mark.asyncio
    async def test_async_iterator(self):
        """Should work as async iterator."""
        config = StreamingResilienceConfig(
            max_retries=3,
            initial_delay_seconds=0.1,
            jitter=False,
        )
        backoff = ExponentialBackoff(config)

        delays = []
        async for delay in backoff:
            delays.append(delay)

        assert len(delays) == 4  # 0 to max_retries inclusive


# =============================================================================
# StreamingCircuitBreaker Tests
# =============================================================================


class TestStreamingCircuitBreaker:
    """Tests for StreamingCircuitBreaker."""

    def test_initial_state_closed(self):
        """Should start in CLOSED state."""
        breaker = StreamingCircuitBreaker("test")

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_open is False

    @pytest.mark.asyncio
    async def test_opens_after_threshold_failures(self):
        """Should open after reaching failure threshold."""
        config = StreamingResilienceConfig(circuit_breaker_threshold=3)
        breaker = StreamingCircuitBreaker("test", config)

        # Record failures
        for i in range(3):
            await breaker.record_failure(Exception(f"Error {i}"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    @pytest.mark.asyncio
    async def test_rejects_calls_when_open(self):
        """Should reject calls when open."""
        config = StreamingResilienceConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_recovery_seconds=10.0,
        )
        breaker = StreamingCircuitBreaker("test", config)

        await breaker.record_failure(Exception("Error"))

        can_execute = await breaker.can_execute()
        assert can_execute is False

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Should transition to HALF_OPEN after recovery timeout."""
        config = StreamingResilienceConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_recovery_seconds=0.1,
        )
        breaker = StreamingCircuitBreaker("test", config)

        await breaker.record_failure(Exception("Error"))
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery
        await asyncio.sleep(0.15)

        can_execute = await breaker.can_execute()
        assert can_execute is True
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_closes_after_successful_half_open_calls(self):
        """Should close after successful calls in HALF_OPEN."""
        config = StreamingResilienceConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_recovery_seconds=0.1,
            circuit_breaker_success_threshold=2,
        )
        breaker = StreamingCircuitBreaker("test", config)

        await breaker.record_failure(Exception("Error"))
        await asyncio.sleep(0.15)
        await breaker.can_execute()  # Transition to HALF_OPEN

        # Record successes
        await breaker.record_success()
        assert breaker.state == CircuitState.HALF_OPEN

        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_half_open_failure(self):
        """Should reopen on failure in HALF_OPEN."""
        config = StreamingResilienceConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_recovery_seconds=0.1,
        )
        breaker = StreamingCircuitBreaker("test", config)

        await breaker.record_failure(Exception("Error"))
        await asyncio.sleep(0.15)
        await breaker.can_execute()  # Transition to HALF_OPEN

        await breaker.record_failure(Exception("Error again"))
        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_context_manager_records_success(self):
        """Should record success when context exits normally."""
        breaker = StreamingCircuitBreaker("test")

        async with await breaker.call():
            pass  # Success

        assert breaker._total_successes == 1

    @pytest.mark.asyncio
    async def test_context_manager_records_failure(self):
        """Should record failure when context exits with exception."""
        breaker = StreamingCircuitBreaker("test")

        with pytest.raises(ValueError):
            async with await breaker.call():
                raise ValueError("Error")

        assert breaker._total_failures == 1

    @pytest.mark.asyncio
    async def test_call_raises_when_open(self):
        """Should raise CircuitBreakerOpenError when open."""
        config = StreamingResilienceConfig(
            circuit_breaker_threshold=1,
            circuit_breaker_recovery_seconds=60.0,
        )
        breaker = StreamingCircuitBreaker("test", config)

        await breaker.record_failure(Exception("Error"))

        with pytest.raises(CircuitBreakerOpenError) as exc_info:
            await breaker.call()

        assert "test" in str(exc_info.value)

    def test_reset(self):
        """Should reset to CLOSED state."""
        breaker = StreamingCircuitBreaker("test")
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 10

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    def test_get_stats(self):
        """Should return statistics."""
        breaker = StreamingCircuitBreaker("test-breaker")

        stats = breaker.get_stats()

        assert stats["name"] == "test-breaker"
        assert stats["state"] == "closed"
        assert "failure_count" in stats
        assert "total_calls" in stats


# =============================================================================
# DLQHandler Tests
# =============================================================================


class TestDLQHandler:
    """Tests for DLQHandler."""

    def test_get_message_id(self):
        """Should generate unique message IDs."""
        handler = DLQHandler()

        id1 = handler.get_message_id("topic1", "key1", offset=100)
        id2 = handler.get_message_id("topic2", "key2", offset=200)

        assert id1 != id2
        assert "topic1" in id1
        assert "100" in id1

    def test_retry_count_tracking(self):
        """Should track retry counts per message."""
        handler = DLQHandler()
        message_id = "test:message:1"

        assert handler.get_retry_count(message_id) == 0

        handler.increment_retry(message_id)
        assert handler.get_retry_count(message_id) == 1

        handler.increment_retry(message_id)
        assert handler.get_retry_count(message_id) == 2

    def test_clear_retry(self):
        """Should clear retry count."""
        handler = DLQHandler()
        message_id = "test:message:1"

        handler.increment_retry(message_id)
        handler.clear_retry(message_id)

        assert handler.get_retry_count(message_id) == 0

    @pytest.mark.asyncio
    async def test_should_send_to_dlq_after_max_retries(self):
        """Should send to DLQ after max retries exceeded."""
        config = StreamingResilienceConfig(dlq_max_retries=2)
        handler = DLQHandler(config)
        message_id = "test:message:1"

        handler.increment_retry(message_id)
        assert await handler.should_send_to_dlq(message_id) is False

        handler.increment_retry(message_id)
        assert await handler.should_send_to_dlq(message_id) is True

    @pytest.mark.asyncio
    async def test_should_not_send_when_dlq_disabled(self):
        """Should not send to DLQ when disabled."""
        config = StreamingResilienceConfig(dlq_enabled=False)
        handler = DLQHandler(config)
        message_id = "test:message:1"

        for _ in range(10):
            handler.increment_retry(message_id)

        assert await handler.should_send_to_dlq(message_id) is False

    @pytest.mark.asyncio
    async def test_handle_failure_returns_false_for_retry(self):
        """Should return False when retrying."""
        config = StreamingResilienceConfig(dlq_max_retries=3)
        handler = DLQHandler(config)

        result = await handler.handle_failure(
            topic="test-topic",
            key="key1",
            value={"data": "test"},
            headers={},
            timestamp=datetime.now(timezone.utc),
            error=Exception("Error"),
            offset=100,
        )

        assert result is False
        assert handler._total_retries == 1

    @pytest.mark.asyncio
    async def test_handle_failure_sends_to_dlq_after_max_retries(self):
        """Should send to DLQ after max retries."""
        config = StreamingResilienceConfig(dlq_max_retries=2)
        dlq_sender = AsyncMock()
        handler = DLQHandler(config, dlq_sender=dlq_sender)

        # First attempt - retry
        result1 = await handler.handle_failure(
            topic="test-topic",
            key="key1",
            value={"data": "test"},
            headers={},
            timestamp=datetime.now(timezone.utc),
            error=Exception("Error"),
            offset=100,
        )
        assert result1 is False

        # Second attempt - DLQ
        result2 = await handler.handle_failure(
            topic="test-topic",
            key="key1",
            value={"data": "test"},
            headers={},
            timestamp=datetime.now(timezone.utc),
            error=Exception("Error"),
            offset=100,
        )
        assert result2 is True
        dlq_sender.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_dlq_calls_sender(self):
        """Should call DLQ sender with correct arguments."""
        dlq_sender = AsyncMock()
        handler = DLQHandler(dlq_sender=dlq_sender)

        await handler.send_to_dlq(
            topic="source-topic",
            key="msg-key",
            value={"data": "test"},
            headers={"header1": "value1"},
            timestamp=datetime.now(timezone.utc),
            error=ValueError("Test error"),
            retry_count=3,
        )

        dlq_sender.assert_called_once()
        call_args = dlq_sender.call_args
        assert call_args[0][0] == "source-topic.dlq"
        assert isinstance(call_args[0][1], DLQMessage)

    @pytest.mark.asyncio
    async def test_on_dlq_send_callback(self):
        """Should call on_dlq_send callback."""
        on_dlq_send = AsyncMock()
        handler = DLQHandler(on_dlq_send=on_dlq_send)

        await handler.send_to_dlq(
            topic="source-topic",
            key="msg-key",
            value={"data": "test"},
            headers={},
            timestamp=datetime.now(timezone.utc),
            error=Exception("Error"),
        )

        on_dlq_send.assert_called_once()

    def test_get_stats(self):
        """Should return statistics."""
        handler = DLQHandler()

        stats = handler.get_stats()

        assert stats["enabled"] is True
        assert "total_messages" in stats
        assert "total_dlq_sends" in stats


# =============================================================================
# DLQMessage Tests
# =============================================================================


class TestDLQMessage:
    """Tests for DLQMessage."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        now = datetime.now(timezone.utc)
        msg = DLQMessage(
            original_topic="test-topic",
            original_key="key1",
            original_value={"data": "test"},
            original_headers={"header1": "value1"},
            original_timestamp=now,
            error_message="Test error",
            error_type="ValueError",
            retry_count=3,
        )

        data = msg.to_dict()

        assert data["original_topic"] == "test-topic"
        assert data["original_key"] == "key1"
        assert data["error_type"] == "ValueError"
        assert data["retry_count"] == 3

    def test_to_json(self):
        """Should serialize to JSON."""
        now = datetime.now(timezone.utc)
        msg = DLQMessage(
            original_topic="test-topic",
            original_key="key1",
            original_value={"data": "test"},
            original_headers={},
            original_timestamp=now,
            error_message="Test error",
            error_type="ValueError",
            retry_count=3,
        )

        json_str = msg.to_json()

        assert isinstance(json_str, str)
        assert "test-topic" in json_str

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        now = datetime.now(timezone.utc)
        data = {
            "original_topic": "test-topic",
            "original_key": "key1",
            "original_value": {"data": "test"},
            "original_headers": {},
            "original_timestamp": now.isoformat(),
            "error_message": "Test error",
            "error_type": "ValueError",
            "retry_count": 3,
            "failed_at": now.isoformat(),
        }

        msg = DLQMessage.from_dict(data)

        assert msg.original_topic == "test-topic"
        assert msg.error_type == "ValueError"


# =============================================================================
# GracefulShutdown Tests
# =============================================================================


class TestGracefulShutdown:
    """Tests for GracefulShutdown."""

    def test_initial_state(self):
        """Should start in non-shutdown state."""
        shutdown = GracefulShutdown()

        assert shutdown.is_shutting_down is False

    def test_register_cleanup(self):
        """Should register cleanup functions."""
        shutdown = GracefulShutdown()

        async def cleanup1():
            pass

        async def cleanup2():
            pass

        shutdown.register_cleanup(cleanup1)
        shutdown.register_cleanup(cleanup2)

        assert len(shutdown._cleanup_tasks) == 2

    def test_trigger_shutdown(self):
        """Should trigger shutdown manually."""
        shutdown = GracefulShutdown()

        shutdown.trigger_shutdown()

        assert shutdown.is_shutting_down is True

    @pytest.mark.asyncio
    async def test_wait_for_shutdown(self):
        """Should wait for shutdown signal."""
        shutdown = GracefulShutdown()

        async def trigger_later():
            await asyncio.sleep(0.1)
            shutdown.trigger_shutdown()

        asyncio.create_task(trigger_later())

        # Should not block forever
        await asyncio.wait_for(shutdown.wait_for_shutdown(), timeout=1.0)
        assert shutdown.is_shutting_down is True


# =============================================================================
# HealthMonitor Tests
# =============================================================================


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.mark.asyncio
    async def test_initial_healthy(self):
        """Should start healthy."""
        monitor = HealthMonitor("test")

        status = await monitor.get_status()

        assert status.healthy is True

    @pytest.mark.asyncio
    async def test_record_success(self):
        """Should update on success."""
        monitor = HealthMonitor("test")

        await monitor.record_success(latency_ms=15.5)

        status = await monitor.get_status()
        assert status.healthy is True
        assert status.latency_ms == 15.5
        assert status.messages_processed == 1

    @pytest.mark.asyncio
    async def test_record_failure(self):
        """Should update on failure."""
        monitor = HealthMonitor("test")

        await monitor.record_failure(Exception("Error"))

        status = await monitor.get_status()
        assert status.messages_failed == 1
        assert status.last_error == "Error"

    @pytest.mark.asyncio
    async def test_unhealthy_after_threshold(self):
        """Should become unhealthy after consecutive failures."""
        config = StreamingResilienceConfig(unhealthy_threshold=3)
        monitor = HealthMonitor("test", config)

        for _ in range(3):
            await monitor.record_failure(Exception("Error"))

        status = await monitor.get_status()
        assert status.healthy is False

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        """Should reset failure count on success."""
        config = StreamingResilienceConfig(unhealthy_threshold=3)
        monitor = HealthMonitor("test", config)

        await monitor.record_failure(Exception("Error"))
        await monitor.record_failure(Exception("Error"))
        await monitor.record_success()

        status = await monitor.get_status()
        assert status.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_reset(self):
        """Should reset to healthy state."""
        monitor = HealthMonitor("test")
        await monitor.record_failure(Exception("Error"))
        await monitor.record_failure(Exception("Error"))
        await monitor.record_failure(Exception("Error"))

        await monitor.reset()

        status = await monitor.get_status()
        assert status.healthy is True


# =============================================================================
# Kafka Connector Resilience Integration Tests
# =============================================================================


class TestKafkaConnectorResilience:
    """Integration tests for Kafka connector resilience."""

    def test_connector_initializes_circuit_breaker(self):
        """Should initialize circuit breaker when enabled."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig(enable_circuit_breaker=True)
        connector = KafkaConnector(config)

        assert connector._circuit_breaker is not None
        assert connector._circuit_breaker.state == CircuitState.CLOSED

    def test_connector_initializes_dlq_handler(self):
        """Should initialize DLQ handler when enabled."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig(enable_dlq=True)
        connector = KafkaConnector(config)

        assert connector._dlq_handler is not None

    def test_connector_initializes_health_monitor(self):
        """Should initialize health monitor."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig()
        connector = KafkaConnector(config)

        assert connector._health_monitor is not None

    def test_connector_initializes_graceful_shutdown(self):
        """Should initialize graceful shutdown when enabled."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig(enable_graceful_shutdown=True)
        connector = KafkaConnector(config)

        assert connector._graceful_shutdown is not None

    def test_connector_can_disable_resilience(self):
        """Should allow disabling resilience components."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig(
            enable_circuit_breaker=False,
            enable_dlq=False,
            enable_graceful_shutdown=False,
        )
        connector = KafkaConnector(config)

        assert connector._circuit_breaker is None
        assert connector._dlq_handler is None
        assert connector._graceful_shutdown is None

    def test_connector_stats_include_resilience(self):
        """Should include resilience stats."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig()
        connector = KafkaConnector(config)

        stats = connector.get_stats()

        assert "circuit_breaker" in stats
        assert "dlq" in stats

    @pytest.mark.asyncio
    async def test_connector_get_health(self):
        """Should return health status."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig()
        connector = KafkaConnector(config)

        health = await connector.get_health()

        assert isinstance(health, HealthStatus)

    def test_connector_reset_circuit_breaker(self):
        """Should reset circuit breaker."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig()
        connector = KafkaConnector(config)
        connector._circuit_breaker._state = CircuitState.OPEN

        connector.reset_circuit_breaker()

        assert connector._circuit_breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_connect_retries_on_failure(self):
        """Should retry connection on failure."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig(
            resilience=StreamingResilienceConfig(
                max_retries=2,
                initial_delay_seconds=0.01,
            ),
        )
        connector = KafkaConnector(config)

        with patch.object(connector, "_connect_internal", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = [
                ConnectionError("Failed 1"),
                ConnectionError("Failed 2"),
                True,
            ]

            result = await connector.connect()

            assert result is True
            assert mock_connect.call_count == 3

    @pytest.mark.asyncio
    async def test_connect_fails_after_max_retries(self):
        """Should fail after max retries exceeded."""
        from aragora.connectors.enterprise.streaming.kafka import (
            KafkaConnector,
            KafkaConfig,
        )

        config = KafkaConfig(
            resilience=StreamingResilienceConfig(
                max_retries=2,
                initial_delay_seconds=0.01,
            ),
        )
        connector = KafkaConnector(config)

        with patch.object(connector, "_connect_internal", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionError("Always fail")

            result = await connector.connect()

            assert result is False
            assert mock_connect.call_count == 3  # Initial + 2 retries
