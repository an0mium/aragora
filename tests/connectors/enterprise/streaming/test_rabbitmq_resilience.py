"""
Tests for RabbitMQ Connector Resilience Patterns.

Tests cover:
- Connection retry with exponential backoff
- Circuit breaker state transitions
- Dead letter queue handling
- Graceful shutdown
- Health monitoring
- Integration with RabbitMQConnector
- Publish resilience

These tests mock the aio-pika library to avoid requiring
an actual RabbitMQ server.
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
    with_retry,
)


# =============================================================================
# RabbitMQ Config Resilience Tests
# =============================================================================


class TestRabbitMQConfigResilience:
    """Tests for RabbitMQConfig resilience settings."""

    def test_default_resilience_config(self):
        """Should include default resilience configuration."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQConfig

        config = RabbitMQConfig(url="amqp://localhost:5672")

        assert config.resilience is not None
        assert config.enable_circuit_breaker is True
        assert config.enable_dlq is True
        assert config.enable_graceful_shutdown is True

    def test_custom_resilience_config(self):
        """Should accept custom resilience configuration."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQConfig

        resilience = StreamingResilienceConfig(
            max_retries=10,
            circuit_breaker_threshold=3,
        )
        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            resilience=resilience,
            enable_circuit_breaker=False,
        )

        assert config.resilience.max_retries == 10
        assert config.enable_circuit_breaker is False


# =============================================================================
# RabbitMQ Connector Resilience Integration Tests
# =============================================================================


class TestRabbitMQConnectorResilience:
    """Integration tests for RabbitMQ connector resilience."""

    def test_connector_initializes_circuit_breaker(self):
        """Should initialize circuit breaker when enabled."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            enable_circuit_breaker=True,
        )
        connector = RabbitMQConnector(config)

        assert connector._circuit_breaker is not None
        assert connector._circuit_breaker.state == CircuitState.CLOSED
        assert connector._circuit_breaker.name == "rabbitmq-broker"

    def test_connector_initializes_dlq_handler(self):
        """Should initialize DLQ handler when enabled."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            enable_dlq=True,
        )
        connector = RabbitMQConnector(config)

        assert connector._dlq_handler is not None

    def test_connector_initializes_health_monitor(self):
        """Should initialize health monitor."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        assert connector._health_monitor is not None
        assert connector._health_monitor.name == "rabbitmq-connector"

    def test_connector_initializes_graceful_shutdown(self):
        """Should initialize graceful shutdown when enabled."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            enable_graceful_shutdown=True,
        )
        connector = RabbitMQConnector(config)

        assert connector._graceful_shutdown is not None

    def test_connector_can_disable_resilience(self):
        """Should allow disabling resilience components."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            enable_circuit_breaker=False,
            enable_dlq=False,
            enable_graceful_shutdown=False,
        )
        connector = RabbitMQConnector(config)

        assert connector._circuit_breaker is None
        assert connector._dlq_handler is None
        assert connector._graceful_shutdown is None

    def test_connector_accepts_custom_dlq_sender(self):
        """Should accept custom DLQ sender."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        custom_sender = AsyncMock()
        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config, dlq_sender=custom_sender)

        assert connector._dlq_handler._dlq_sender == custom_sender

    def test_connector_stats_include_resilience(self):
        """Should include resilience stats."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        stats = connector.get_stats()

        assert "circuit_breaker" in stats
        assert "dlq" in stats
        assert "dlq_count" in stats

    def test_connector_stats_hide_credentials(self):
        """Should hide credentials in stats."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://user:secret@localhost:5672")
        connector = RabbitMQConnector(config)

        stats = connector.get_stats()

        # URL should not contain password
        assert "secret" not in stats["url"]
        assert "localhost" in stats["url"]

    @pytest.mark.asyncio
    async def test_connector_get_health(self):
        """Should return health status."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        health = await connector.get_health()

        assert isinstance(health, HealthStatus)
        assert health.healthy is True

    def test_connector_reset_circuit_breaker(self):
        """Should reset circuit breaker."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)
        connector._circuit_breaker._state = CircuitState.OPEN

        connector.reset_circuit_breaker()

        assert connector._circuit_breaker.state == CircuitState.CLOSED


# =============================================================================
# RabbitMQ Connection Retry Tests
# =============================================================================


class TestRabbitMQConnectionRetry:
    """Tests for RabbitMQ connection retry behavior."""

    @pytest.mark.asyncio
    async def test_connect_retries_on_failure(self):
        """Should retry connection on failure."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            resilience=StreamingResilienceConfig(
                max_retries=2,
                initial_delay_seconds=0.01,
            ),
        )
        connector = RabbitMQConnector(config)

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
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            resilience=StreamingResilienceConfig(
                max_retries=2,
                initial_delay_seconds=0.01,
            ),
        )
        connector = RabbitMQConnector(config)

        with patch.object(connector, "_connect_internal", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionError("Always fail")

            result = await connector.connect()

            assert result is False
            assert mock_connect.call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_connect_skips_when_circuit_open(self):
        """Should skip connection when circuit breaker is open."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            resilience=StreamingResilienceConfig(
                circuit_breaker_threshold=1,
                circuit_breaker_recovery_seconds=60.0,
            ),
        )
        connector = RabbitMQConnector(config)

        # Open the circuit breaker
        await connector._circuit_breaker.record_failure(Exception("Error"))

        with patch.object(connector, "_connect_internal", new_callable=AsyncMock) as mock_connect:
            result = await connector.connect()

            assert result is False
            mock_connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_records_success(self):
        """Should record success on successful connection."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        with patch.object(
            connector, "_connect_internal", new_callable=AsyncMock, return_value=True
        ):
            await connector.connect()

            assert connector._circuit_breaker._total_successes == 1
            status = await connector._health_monitor.get_status()
            assert status.messages_processed == 1

    @pytest.mark.asyncio
    async def test_connect_records_failure(self):
        """Should record failure on connection error."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            resilience=StreamingResilienceConfig(
                max_retries=0,
                initial_delay_seconds=0.01,
            ),
        )
        connector = RabbitMQConnector(config)

        with patch.object(connector, "_connect_internal", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionError("Failed")

            await connector.connect()

            assert connector._circuit_breaker._total_failures == 1


# =============================================================================
# RabbitMQ Publish Resilience Tests
# =============================================================================


class TestRabbitMQPublishResilience:
    """Tests for RabbitMQ publish resilience."""

    @pytest.mark.asyncio
    async def test_publish_checks_circuit_breaker(self):
        """Should check circuit breaker before publishing."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            resilience=StreamingResilienceConfig(
                circuit_breaker_threshold=1,
                circuit_breaker_recovery_seconds=60.0,
            ),
        )
        connector = RabbitMQConnector(config)

        # Open the circuit breaker
        await connector._circuit_breaker.record_failure(Exception("Error"))

        with pytest.raises(CircuitBreakerOpenError):
            await connector.publish({"data": "test"})

    @pytest.mark.asyncio
    async def test_publish_records_success(self):
        """Should record success on successful publish."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        # Mock the channel and aio_pika module
        mock_channel = MagicMock()
        mock_channel.default_exchange = MagicMock()
        mock_channel.default_exchange.publish = AsyncMock()
        connector._channel = mock_channel

        # Mock aio_pika at import time
        import sys

        mock_aio_pika = MagicMock()
        mock_aio_pika.Message = MagicMock(return_value=MagicMock())
        mock_aio_pika.DeliveryMode = MagicMock()
        mock_aio_pika.DeliveryMode.PERSISTENT = 2
        mock_aio_pika.DeliveryMode.NOT_PERSISTENT = 1

        with patch.dict(sys.modules, {"aio_pika": mock_aio_pika}):
            result = await connector.publish({"data": "test"})

            assert result is True
            assert connector._circuit_breaker._total_successes == 1

    @pytest.mark.asyncio
    async def test_publish_records_failure(self):
        """Should record failure on publish error."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        # Mock channel to raise error
        mock_channel = MagicMock()
        mock_channel.default_exchange = MagicMock()
        mock_channel.default_exchange.publish = AsyncMock(side_effect=ConnectionError("Publish failed"))
        connector._channel = mock_channel

        # Mock aio_pika at import time
        import sys

        mock_aio_pika = MagicMock()
        mock_aio_pika.Message = MagicMock(return_value=MagicMock())
        mock_aio_pika.DeliveryMode = MagicMock()
        mock_aio_pika.DeliveryMode.PERSISTENT = 2
        mock_aio_pika.DeliveryMode.NOT_PERSISTENT = 1

        with patch.dict(sys.modules, {"aio_pika": mock_aio_pika}):
            result = await connector.publish({"data": "test"})

            assert result is False
            assert connector._circuit_breaker._total_failures == 1


# =============================================================================
# RabbitMQ DLQ Sender Tests
# =============================================================================


class TestRabbitMQDLQSender:
    """Tests for RabbitMQ default DLQ sender."""

    @pytest.mark.asyncio
    async def test_default_dlq_sender_warns_without_channel(self):
        """Should warn when no channel available."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        # No channel set
        msg = DLQMessage(
            original_topic="test-queue",
            original_key="key1",
            original_value={"data": "test"},
            original_headers={},
            original_timestamp=datetime.now(timezone.utc),
            error_message="Test error",
            error_type="ValueError",
            retry_count=3,
        )

        # Should not raise, just log warning
        await connector._default_dlq_sender("test-queue.dlq", msg)


# =============================================================================
# with_retry Decorator Tests
# =============================================================================


class TestWithRetryDecorator:
    """Tests for the with_retry decorator."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self):
        """Should return result on first success."""
        config = StreamingResilienceConfig(max_retries=3)

        @with_retry(config)
        async def succeed():
            return "success"

        result = await succeed()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        """Should retry on retryable exception."""
        config = StreamingResilienceConfig(
            max_retries=3,
            initial_delay_seconds=0.01,
        )
        call_count = 0

        @with_retry(config, retryable_exceptions=(ConnectionError,))
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Failed")
            return "success"

        result = await fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self):
        """Should raise after max retries exceeded."""
        config = StreamingResilienceConfig(
            max_retries=2,
            initial_delay_seconds=0.01,
        )

        @with_retry(config, retryable_exceptions=(ConnectionError,))
        async def always_fail():
            raise ConnectionError("Always fail")

        with pytest.raises(ConnectionError):
            await always_fail()

    @pytest.mark.asyncio
    async def test_does_not_retry_non_retryable(self):
        """Should not retry non-retryable exceptions."""
        config = StreamingResilienceConfig(max_retries=3)
        call_count = 0

        @with_retry(config, retryable_exceptions=(ConnectionError,))
        async def raise_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await raise_value_error()

        assert call_count == 1


# =============================================================================
# RabbitMQ Graceful Shutdown Integration Tests
# =============================================================================


class TestRabbitMQGracefulShutdown:
    """Tests for RabbitMQ graceful shutdown integration."""

    @pytest.mark.asyncio
    async def test_start_registers_cleanup(self):
        """Should register cleanup on start."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            enable_graceful_shutdown=True,
        )
        connector = RabbitMQConnector(config)

        with patch.object(connector, "connect", new_callable=AsyncMock, return_value=True):
            with patch.object(connector._graceful_shutdown, "setup_signal_handlers") as mock_setup:
                await connector.start()

                mock_setup.assert_called_once()
                assert len(connector._graceful_shutdown._cleanup_tasks) == 1

    @pytest.mark.asyncio
    async def test_cleanup_disconnects(self):
        """Should disconnect on cleanup."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        with patch.object(connector, "disconnect", new_callable=AsyncMock) as mock_disconnect:
            await connector._cleanup()

            mock_disconnect.assert_called_once()


# =============================================================================
# RabbitMQ Health Status Tests
# =============================================================================


class TestRabbitMQHealthStatus:
    """Tests for RabbitMQ health status."""

    @pytest.mark.asyncio
    async def test_health_reflects_running_state(self):
        """Should reflect running state in health."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        # Initially not running
        health = await connector.get_health()
        assert health.healthy is True  # Health monitor starts healthy

        # After recording failures
        await connector._health_monitor.record_failure(Exception("Error"))
        await connector._health_monitor.record_failure(Exception("Error"))
        await connector._health_monitor.record_failure(Exception("Error"))

        health = await connector.get_health()
        assert health.healthy is False

    def test_health_status_to_dict(self):
        """Should serialize health status to dict."""
        status = HealthStatus(
            healthy=True,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=0,
            latency_ms=15.5,
            messages_processed=100,
            messages_failed=2,
        )

        data = status.to_dict()

        assert data["healthy"] is True
        assert data["latency_ms"] == 15.5
        assert data["messages_processed"] == 100


# =============================================================================
# RabbitMQ Circuit Breaker Integration Tests
# =============================================================================


class TestRabbitMQCircuitBreakerIntegration:
    """Integration tests for circuit breaker with RabbitMQ connector."""

    @pytest.mark.asyncio
    async def test_circuit_opens_after_connection_failures(self):
        """Should open circuit after connection failures."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://localhost:5672",
            resilience=StreamingResilienceConfig(
                max_retries=0,
                circuit_breaker_threshold=3,
                initial_delay_seconds=0.01,
            ),
        )
        connector = RabbitMQConnector(config)

        with patch.object(connector, "_connect_internal", new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionError("Failed")

            # Multiple connection attempts
            for _ in range(3):
                await connector.connect()

            assert connector._circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_in_stats(self):
        """Should include circuit breaker state in stats."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(url="amqp://localhost:5672")
        connector = RabbitMQConnector(config)

        stats = connector.get_stats()

        assert stats["circuit_breaker"]["state"] == "closed"
        assert "failure_count" in stats["circuit_breaker"]
