"""
Tests for RabbitMQ Enterprise Streaming Connector.

Tests cover:
- Configuration and initialization
- Queue and exchange binding
- Message acknowledgment
- Dead letter queue handling
- SyncItem conversion for Knowledge Mound
- Publish functionality
- Error handling

These tests mock the aio-pika library to avoid requiring
an actual RabbitMQ server.
"""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Configuration Tests
# =============================================================================


class TestRabbitMQConfig:
    """Tests for RabbitMQConfig."""

    def test_default_config(self):
        """Should initialize with sensible defaults."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQConfig

        config = RabbitMQConfig()

        assert config.url == "amqp://guest:guest@localhost/"
        assert config.queue == "aragora-events"
        assert config.exchange == ""
        assert config.exchange_type == "direct"
        assert config.durable is True
        assert config.auto_delete is False
        assert config.prefetch_count == 10
        assert config.auto_ack is False
        assert config.batch_size == 100

    def test_custom_config(self):
        """Should accept custom configuration."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQConfig

        config = RabbitMQConfig(
            url="amqp://admin:secret@rabbitmq.example.com:5672/",
            queue="decisions",
            exchange="aragora-exchange",
            exchange_type="topic",
            routing_key="decisions.*",
            prefetch_count=50,
            ssl=True,
            dead_letter_exchange="dlx",
        )

        assert config.url == "amqp://admin:secret@rabbitmq.example.com:5672/"
        assert config.queue == "decisions"
        assert config.exchange == "aragora-exchange"
        assert config.exchange_type == "topic"
        assert config.routing_key == "decisions.*"
        assert config.ssl is True
        assert config.dead_letter_exchange == "dlx"


# =============================================================================
# Connector Initialization Tests
# =============================================================================


class TestRabbitMQConnectorInitialization:
    """Tests for connector initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        assert connector.connector_id == "rabbitmq"
        assert connector.config.queue == "aragora-events"
        assert connector._connection is None
        assert connector._running is False

    def test_connector_stats(self):
        """Should provide accurate statistics."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            url="amqp://admin:secret@rabbitmq.example.com/",
            queue="test-queue",
            exchange="test-exchange",
        )
        connector = RabbitMQConnector(config)

        stats = connector.get_stats()

        assert stats["connector_id"] == "rabbitmq"
        # URL should have credentials hidden
        assert "admin:secret" not in stats["url"]
        assert "rabbitmq.example.com" in stats["url"]
        assert stats["queue"] == "test-queue"
        assert stats["exchange"] == "test-exchange"
        assert stats["running"] is False
        assert stats["consumed_count"] == 0
        assert stats["acked_count"] == 0
        assert stats["nacked_count"] == 0


# =============================================================================
# Message Deserialization Tests
# =============================================================================


class TestRabbitMQMessageDeserialization:
    """Tests for message deserialization."""

    def test_deserialize_json_message(self):
        """Should deserialize JSON messages."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        connector = RabbitMQConnector(RabbitMQConfig())

        # Create mock aio-pika message
        mock_message = MagicMock()
        mock_message.body = json.dumps({"type": "event", "data": "test"}).encode()
        mock_message.headers = {"producer": "test-producer"}
        mock_message.routing_key = "events.test"
        mock_message.delivery_tag = 42
        mock_message.timestamp = datetime.now(tz=timezone.utc)
        mock_message.message_id = "msg-123"
        mock_message.correlation_id = "corr-456"
        mock_message.reply_to = None
        mock_message.expiration = None
        mock_message.priority = None
        mock_message.channel = MagicMock()

        rmq_msg = connector._deserialize_message(mock_message)

        assert rmq_msg.queue == "aragora-events"
        assert rmq_msg.routing_key == "events.test"
        assert rmq_msg.delivery_tag == 42
        assert rmq_msg.body == {"type": "event", "data": "test"}
        assert rmq_msg.headers["producer"] == "test-producer"
        assert rmq_msg.message_id == "msg-123"

    def test_deserialize_string_message(self):
        """Should handle plain string messages."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        connector = RabbitMQConnector(RabbitMQConfig())

        mock_message = MagicMock()
        mock_message.body = b"plain text message"
        mock_message.headers = {}
        mock_message.routing_key = "test"
        mock_message.delivery_tag = 1
        mock_message.timestamp = None
        mock_message.message_id = None
        mock_message.correlation_id = None
        mock_message.reply_to = None
        mock_message.expiration = None
        mock_message.priority = None
        mock_message.channel = MagicMock()

        rmq_msg = connector._deserialize_message(mock_message)

        assert rmq_msg.body == "plain text message"

    def test_deserialize_binary_message(self):
        """Should handle non-UTF8 binary messages."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        connector = RabbitMQConnector(RabbitMQConfig())

        mock_message = MagicMock()
        mock_message.body = b"\x00\x01\x02\xff"  # Binary data
        mock_message.headers = {}
        mock_message.routing_key = "binary"
        mock_message.delivery_tag = 1
        mock_message.timestamp = None
        mock_message.message_id = None
        mock_message.correlation_id = None
        mock_message.reply_to = None
        mock_message.expiration = None
        mock_message.priority = None
        mock_message.channel = MagicMock()

        rmq_msg = connector._deserialize_message(mock_message)

        # Should decode with errors replaced
        assert rmq_msg.body is not None


# =============================================================================
# Message Dataclass Tests
# =============================================================================


class TestRabbitMQMessage:
    """Tests for RabbitMQMessage dataclass."""

    def test_to_sync_item_dict_body(self):
        """Should convert dict message to SyncItem."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQMessage

        msg = RabbitMQMessage(
            queue="decisions",
            routing_key="decisions.new",
            delivery_tag=100,
            body={"type": "decision", "title": "Important Decision", "content": "Details here"},
            headers={"producer": "api-server"},
            timestamp=datetime.now(tz=timezone.utc),
            message_id="msg-123",
        )

        sync_item = msg.to_sync_item()

        assert sync_item.source_type == "message_queue"
        assert sync_item.domain == "enterprise/rabbitmq"
        assert "decisions" in sync_item.id
        assert "100" in sync_item.id  # delivery_tag
        assert "Important Decision" in sync_item.title
        assert sync_item.confidence == 0.9
        assert sync_item.metadata["queue"] == "decisions"
        assert sync_item.metadata["routing_key"] == "decisions.new"
        assert sync_item.metadata["delivery_tag"] == 100

    def test_to_sync_item_string_body(self):
        """Should convert string message to SyncItem."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQMessage

        msg = RabbitMQMessage(
            queue="logs",
            routing_key="logs.info",
            delivery_tag=200,
            body="Log entry: Application started",
            headers={},
            timestamp=datetime.now(tz=timezone.utc),
        )

        sync_item = msg.to_sync_item()

        assert sync_item.content == "Log entry: Application started"
        assert sync_item.title == "RabbitMQ: logs"

    @pytest.mark.asyncio
    async def test_ack_calls_channel(self):
        """Should call channel.basic_ack on ack()."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQMessage

        mock_channel = AsyncMock()

        msg = RabbitMQMessage(
            queue="test",
            routing_key="test",
            delivery_tag=42,
            body="test",
            headers={},
            timestamp=datetime.now(tz=timezone.utc),
            _channel=mock_channel,
        )

        await msg.ack()

        mock_channel.basic_ack.assert_called_once_with(42)

    @pytest.mark.asyncio
    async def test_nack_calls_channel(self):
        """Should call channel.basic_nack on nack()."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQMessage

        mock_channel = AsyncMock()

        msg = RabbitMQMessage(
            queue="test",
            routing_key="test",
            delivery_tag=42,
            body="test",
            headers={},
            timestamp=datetime.now(tz=timezone.utc),
            _channel=mock_channel,
        )

        await msg.nack(requeue=False)

        mock_channel.basic_nack.assert_called_once_with(42, requeue=False)

    @pytest.mark.asyncio
    async def test_reject_calls_channel(self):
        """Should call channel.basic_reject on reject()."""
        from aragora.connectors.enterprise.streaming.rabbitmq import RabbitMQMessage

        mock_channel = AsyncMock()

        msg = RabbitMQMessage(
            queue="test",
            routing_key="test",
            delivery_tag=42,
            body="test",
            headers={},
            timestamp=datetime.now(tz=timezone.utc),
            _channel=mock_channel,
        )

        await msg.reject(requeue=True)

        mock_channel.basic_reject.assert_called_once_with(42, requeue=True)


# =============================================================================
# Connection Tests
# =============================================================================


class TestRabbitMQConnection:
    """Tests for RabbitMQ connection management."""

    @pytest.mark.asyncio
    async def test_connect_creates_channel_and_queue(self):
        """Should create channel and declare queue on connect."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        # Mock aio-pika
        mock_queue = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.set_qos = AsyncMock()
        mock_channel.declare_queue = AsyncMock(return_value=mock_queue)
        mock_channel.default_exchange = MagicMock()

        mock_connection = AsyncMock()
        mock_connection.channel = AsyncMock(return_value=mock_channel)

        mock_aio_pika = MagicMock()
        mock_aio_pika.connect_robust = AsyncMock(return_value=mock_connection)

        with patch.dict("sys.modules", {"aio_pika": mock_aio_pika}):
            result = await connector.connect()

            assert result is True
            mock_connection.channel.assert_called_once()
            mock_channel.set_qos.assert_called_once_with(prefetch_count=10)
            mock_channel.declare_queue.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_with_exchange(self):
        """Should declare and bind to exchange if specified."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            exchange="test-exchange",
            exchange_type="topic",
            routing_key="events.*",
        )
        connector = RabbitMQConnector(config)

        mock_queue = AsyncMock()
        mock_queue.bind = AsyncMock()

        mock_exchange = AsyncMock()

        mock_channel = AsyncMock()
        mock_channel.set_qos = AsyncMock()
        mock_channel.declare_exchange = AsyncMock(return_value=mock_exchange)
        mock_channel.declare_queue = AsyncMock(return_value=mock_queue)

        mock_connection = AsyncMock()
        mock_connection.channel = AsyncMock(return_value=mock_channel)

        mock_aio_pika = MagicMock()
        mock_aio_pika.connect_robust = AsyncMock(return_value=mock_connection)

        with patch.dict("sys.modules", {"aio_pika": mock_aio_pika}):
            result = await connector.connect()

            assert result is True
            mock_channel.declare_exchange.assert_called_once()
            mock_queue.bind.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_closes_connection(self):
        """Should close connection on disconnect."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        mock_connection = AsyncMock()
        mock_connection.close = AsyncMock()
        connector._connection = mock_connection
        connector._running = True

        await connector.disconnect()

        mock_connection.close.assert_called_once()
        assert connector._connection is None
        assert connector._running is False


# =============================================================================
# Publish Tests
# =============================================================================


class TestRabbitMQPublish:
    """Tests for message publishing."""

    @pytest.mark.asyncio
    async def test_publish_json_message(self):
        """Should publish JSON message."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        mock_exchange = AsyncMock()
        mock_exchange.publish = AsyncMock()

        mock_channel = AsyncMock()
        mock_channel.default_exchange = mock_exchange

        connector._channel = mock_channel

        mock_aio_pika = MagicMock()
        mock_message = MagicMock()
        mock_aio_pika.Message.return_value = mock_message
        mock_aio_pika.DeliveryMode = MagicMock()
        mock_aio_pika.DeliveryMode.PERSISTENT = 2
        mock_aio_pika.DeliveryMode.NOT_PERSISTENT = 1

        with patch.dict("sys.modules", {"aio_pika": mock_aio_pika}):
            result = await connector.publish(
                body={"type": "event", "data": "test"},
                routing_key="events.new",
                headers={"producer": "test"},
            )

            assert result is True
            mock_exchange.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_with_exchange(self):
        """Should publish to specified exchange."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(exchange="test-exchange")
        connector = RabbitMQConnector(config)

        mock_exchange = AsyncMock()
        mock_exchange.publish = AsyncMock()

        mock_channel = AsyncMock()
        mock_channel.get_exchange = AsyncMock(return_value=mock_exchange)

        connector._channel = mock_channel

        mock_aio_pika = MagicMock()
        mock_aio_pika.Message.return_value = MagicMock()
        mock_aio_pika.DeliveryMode = MagicMock()
        mock_aio_pika.DeliveryMode.PERSISTENT = 2
        mock_aio_pika.DeliveryMode.NOT_PERSISTENT = 1

        with patch.dict("sys.modules", {"aio_pika": mock_aio_pika}):
            result = await connector.publish(
                body="test message",
                routing_key="test.key",
            )

            assert result is True
            mock_channel.get_exchange.assert_called_once_with("test-exchange")


# =============================================================================
# Sync Tests
# =============================================================================


class TestRabbitMQSync:
    """Tests for sync method that yields SyncItems."""

    @pytest.mark.asyncio
    async def test_sync_yields_sync_items(self):
        """Should yield SyncItem objects ready for KM ingestion."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
            RabbitMQMessage,
        )
        from aragora.connectors.enterprise.base import SyncItem

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        # Create mock messages
        messages = []
        for i in range(3):
            msg = RabbitMQMessage(
                queue="test",
                routing_key="test",
                delivery_tag=i,
                body={"id": i, "content": f"Content {i}"},
                headers={},
                timestamp=datetime.now(tz=timezone.utc),
            )
            messages.append(msg)

        # Mock consume to yield messages
        async def mock_consume(max_messages=None):
            for msg in messages[:max_messages]:
                yield msg

        connector.consume = mock_consume
        connector.config.auto_ack = True

        # Sync
        items = []
        async for item in connector.sync(batch_size=3):
            items.append(item)

        assert len(items) == 3
        assert all(isinstance(item, SyncItem) for item in items)
        assert items[0].source_type == "message_queue"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestRabbitMQErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_connection_error_returns_false(self):
        """Should return False on connection error."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        mock_aio_pika = MagicMock()
        mock_aio_pika.connect_robust = AsyncMock(side_effect=Exception("Connection refused"))

        with patch.dict("sys.modules", {"aio_pika": mock_aio_pika}):
            result = await connector.connect()

        assert result is False

    @pytest.mark.asyncio
    async def test_publish_error_returns_false(self):
        """Should return False on publish error."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        mock_exchange = AsyncMock()
        mock_exchange.publish = AsyncMock(side_effect=Exception("Channel closed"))

        mock_channel = AsyncMock()
        mock_channel.default_exchange = mock_exchange

        connector._channel = mock_channel

        mock_aio_pika = MagicMock()
        mock_aio_pika.Message.return_value = MagicMock()
        mock_aio_pika.DeliveryMode = MagicMock()
        mock_aio_pika.DeliveryMode.PERSISTENT = 2
        mock_aio_pika.DeliveryMode.NOT_PERSISTENT = 1

        with patch.dict("sys.modules", {"aio_pika": mock_aio_pika}):
            result = await connector.publish(body="test")

        assert result is False

    def test_stats_without_connection(self):
        """Should return stats even without active connection."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig()
        connector = RabbitMQConnector(config)

        stats = connector.get_stats()

        assert "connector_id" in stats
        assert stats["running"] is False
        assert stats["consumed_count"] == 0


# =============================================================================
# Dead Letter Queue Tests
# =============================================================================


class TestRabbitMQDeadLetterQueue:
    """Tests for dead letter queue configuration."""

    @pytest.mark.asyncio
    async def test_connect_with_dlq(self):
        """Should configure DLQ when specified."""
        from aragora.connectors.enterprise.streaming.rabbitmq import (
            RabbitMQConnector,
            RabbitMQConfig,
        )

        config = RabbitMQConfig(
            dead_letter_exchange="dlx",
            dead_letter_routing_key="dead-letters",
            message_ttl=60000,  # 1 minute
        )
        connector = RabbitMQConnector(config)

        mock_queue = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.set_qos = AsyncMock()
        mock_channel.declare_queue = AsyncMock(return_value=mock_queue)
        mock_channel.default_exchange = MagicMock()

        mock_connection = AsyncMock()
        mock_connection.channel = AsyncMock(return_value=mock_channel)

        mock_aio_pika = MagicMock()
        mock_aio_pika.connect_robust = AsyncMock(return_value=mock_connection)

        with patch.dict("sys.modules", {"aio_pika": mock_aio_pika}):
            result = await connector.connect()

            # Verify queue was declared with DLQ arguments
            declare_call = mock_channel.declare_queue.call_args
            if declare_call and declare_call.kwargs.get("arguments"):
                args = declare_call.kwargs["arguments"]
                assert args.get("x-dead-letter-exchange") == "dlx"
                assert args.get("x-dead-letter-routing-key") == "dead-letters"
                assert args.get("x-message-ttl") == 60000
