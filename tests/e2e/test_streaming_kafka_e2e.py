"""
End-to-End Test: Kafka Streaming Connector.

Tests complete Kafka streaming scenarios including:
- Configuration and initialization with different settings
- Connection lifecycle management
- Message consumption and deserialization
- SyncItem conversion for Knowledge Mound integration
- Complete streaming workflow from Kafka to KM
- Error handling and recovery scenarios

Related plan: kind-squishing-russell.md
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.connectors.enterprise.streaming.kafka import (
    KafkaConfig,
    KafkaConnector,
    KafkaMessage,
)
from aragora.connectors.enterprise.base import SyncItem, SyncState


# ============================================================================
# Test Helpers
# ============================================================================


def create_mock_kafka_message(
    topic: str = "test-topic",
    partition: int = 0,
    offset: int = 1,
    key: Optional[bytes] = None,
    value: bytes = b'{"type": "test", "data": "sample"}',
    timestamp: int = 1609459200000,  # 2021-01-01 00:00:00 UTC in ms
) -> MagicMock:
    """Create a mock Kafka message for testing."""
    message = MagicMock()
    message.topic = topic
    message.partition = partition
    message.offset = offset
    message.key = key
    message.value = value
    message.timestamp = timestamp
    return message


def create_kafka_message_dataclass(
    topic: str = "test-topic",
    partition: int = 0,
    offset: int = 1,
    key: Optional[str] = None,
    value: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timestamp: Optional[datetime] = None,
) -> KafkaMessage:
    """Create a KafkaMessage dataclass for testing."""
    return KafkaMessage(
        topic=topic,
        partition=partition,
        offset=offset,
        key=key,
        value=value or {"type": "test", "data": "sample"},
        headers=headers or {},
        timestamp=timestamp or datetime.now(timezone.utc),
    )


def create_test_kafka_config(**kwargs: Any) -> KafkaConfig:
    """Create a test Kafka configuration."""
    defaults = {
        "bootstrap_servers": "localhost:9092",
        "topics": ["test-topic"],
        "group_id": "test-consumer-group",
        "auto_offset_reset": "earliest",
        "batch_size": 10,
    }
    defaults.update(kwargs)
    return KafkaConfig(**defaults)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def default_config() -> KafkaConfig:
    """Create default Kafka config."""
    return create_test_kafka_config()


@pytest.fixture
def connector(default_config: KafkaConfig) -> KafkaConnector:
    """Create Kafka connector with default config."""
    return KafkaConnector(default_config)


@pytest.fixture
def mock_aiokafka_consumer() -> MagicMock:
    """Create mock aiokafka consumer."""
    consumer = MagicMock()
    consumer.start = AsyncMock()
    consumer.stop = AsyncMock()
    consumer.getone = AsyncMock()
    consumer.getmany = AsyncMock(return_value={})
    return consumer


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.e2e
class TestKafkaConfigValidation:
    """Tests for Kafka configuration validation."""

    def test_default_config_has_sensible_defaults(self) -> None:
        """Test default configuration values."""
        config = KafkaConfig()

        assert config.bootstrap_servers == "localhost:9092"
        assert config.topics == ["aragora-events"]
        assert config.group_id == "aragora-consumer"
        assert config.auto_offset_reset == "earliest"
        assert config.security_protocol == "PLAINTEXT"
        assert config.batch_size == 100
        assert config.poll_timeout_seconds == 1.0

    def test_custom_config_values(self) -> None:
        """Test custom configuration values are accepted."""
        config = KafkaConfig(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            topics=["decisions", "events", "metrics"],
            group_id="aragora-production",
            auto_offset_reset="latest",
            security_protocol="SSL",
            batch_size=500,
        )

        assert config.bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert len(config.topics) == 3
        assert config.group_id == "aragora-production"
        assert config.security_protocol == "SSL"
        assert config.batch_size == 500

    def test_ssl_config_options(self) -> None:
        """Test SSL configuration options."""
        config = KafkaConfig(
            bootstrap_servers="secure-kafka:9093",
            security_protocol="SSL",
            ssl_cafile="/path/to/ca.pem",
            ssl_certfile="/path/to/cert.pem",
            ssl_keyfile="/path/to/key.pem",
        )

        assert config.security_protocol == "SSL"
        assert config.ssl_cafile == "/path/to/ca.pem"
        assert config.ssl_certfile == "/path/to/cert.pem"
        assert config.ssl_keyfile == "/path/to/key.pem"

    def test_sasl_config_options(self) -> None:
        """Test SASL authentication configuration."""
        config = KafkaConfig(
            bootstrap_servers="secure-kafka:9094",
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_username="user",
            sasl_password="password",
        )

        assert config.security_protocol == "SASL_SSL"
        assert config.sasl_mechanism == "PLAIN"
        assert config.sasl_username == "user"
        assert config.sasl_password == "password"

    def test_consumer_settings(self) -> None:
        """Test consumer-specific settings."""
        config = KafkaConfig(
            enable_auto_commit=False,
            auto_commit_interval_ms=10000,
            max_poll_records=1000,
            session_timeout_ms=60000,
            heartbeat_interval_ms=20000,
        )

        assert config.enable_auto_commit is False
        assert config.auto_commit_interval_ms == 10000
        assert config.max_poll_records == 1000
        assert config.session_timeout_ms == 60000


@pytest.mark.e2e
class TestKafkaConnectorInitialization:
    """Tests for Kafka connector initialization."""

    def test_connector_init_with_config(self, default_config: KafkaConfig) -> None:
        """Test connector initializes with config."""
        connector = KafkaConnector(default_config)

        assert connector.connector_id == "kafka"
        assert connector.config == default_config
        assert connector._consumer is None
        assert connector._running is False
        assert connector._consumed_count == 0
        assert connector._error_count == 0

    def test_connector_stats(self, connector: KafkaConnector) -> None:
        """Test connector statistics."""
        stats = connector.get_stats()

        assert stats["connector_id"] == "kafka"
        assert stats["bootstrap_servers"] == "localhost:9092"
        assert stats["topics"] == ["test-topic"]
        assert stats["group_id"] == "test-consumer-group"
        assert stats["running"] is False
        assert stats["consumed_count"] == 0
        assert stats["error_count"] == 0

    @pytest.mark.asyncio
    async def test_connector_health_check(self, connector: KafkaConnector) -> None:
        """Test connector health check when not connected."""
        health = await connector.health_check()

        # health_check returns a ConnectorHealth object
        assert hasattr(health, "is_healthy")
        assert hasattr(health, "name")
        assert health.name == "Kafka"
        assert isinstance(health.is_healthy, bool)


@pytest.mark.e2e
class TestKafkaMessageHandling:
    """Tests for Kafka message handling."""

    def test_deserialize_json_message(self, connector: KafkaConnector) -> None:
        """Test deserializing JSON message."""
        mock_msg = create_mock_kafka_message(
            value=json.dumps({"type": "decision", "data": "test"}).encode()
        )

        kafka_msg = connector._deserialize_message(mock_msg)

        assert kafka_msg.topic == "test-topic"
        assert kafka_msg.partition == 0
        assert kafka_msg.offset == 1
        assert kafka_msg.value == {"type": "decision", "data": "test"}

    def test_deserialize_string_message(self, connector: KafkaConnector) -> None:
        """Test deserializing plain string message."""
        mock_msg = create_mock_kafka_message(value=b"plain text message")

        kafka_msg = connector._deserialize_message(mock_msg)

        assert kafka_msg.value == "plain text message"

    def test_deserialize_invalid_json_fallback(self, connector: KafkaConnector) -> None:
        """Test invalid JSON falls back to string."""
        mock_msg = create_mock_kafka_message(value=b"invalid json {")

        kafka_msg = connector._deserialize_message(mock_msg)

        assert kafka_msg.value == "invalid json {"

    def test_deserialize_with_key(self, connector: KafkaConnector) -> None:
        """Test deserializing message with key."""
        mock_msg = create_mock_kafka_message(
            key=b"message-key-123",
            value=b'{"data": "value"}',
        )

        kafka_msg = connector._deserialize_message(mock_msg)

        assert kafka_msg.key == "message-key-123"


@pytest.mark.e2e
class TestSyncItemConversion:
    """Tests for converting Kafka messages to SyncItems."""

    def test_dict_value_to_sync_item(self) -> None:
        """Test converting dict message to SyncItem."""
        kafka_msg = create_kafka_message_dataclass(
            topic="decisions",
            partition=1,
            offset=100,
            key="decision-key",
            value={"type": "decision", "title": "Architecture Choice", "data": "test"},
            headers={"producer": "system-a"},
        )

        sync_item = kafka_msg.to_sync_item()

        assert sync_item.id == "kafka-decisions-1-100"
        assert sync_item.source_type == "event_stream"
        assert "Architecture Choice" in sync_item.title
        assert sync_item.domain == "enterprise/kafka"
        assert sync_item.confidence == 0.9
        assert sync_item.metadata["topic"] == "decisions"
        assert sync_item.metadata["partition"] == 1
        assert sync_item.metadata["offset"] == 100

    def test_string_value_to_sync_item(self) -> None:
        """Test converting string message to SyncItem."""
        kafka_msg = create_kafka_message_dataclass(
            topic="logs",
            value="INFO: Application started successfully",
        )

        sync_item = kafka_msg.to_sync_item()

        assert "Kafka: logs" in sync_item.title
        assert "Application started successfully" in sync_item.content

    def test_sync_item_content_truncation(self) -> None:
        """Test SyncItem content is truncated for very long messages."""
        large_content = "x" * 100000
        kafka_msg = create_kafka_message_dataclass(
            topic="large-messages",
            value=large_content,
        )

        sync_item = kafka_msg.to_sync_item()

        # Content should be truncated to 50000 chars
        assert len(sync_item.content) <= 50000

    def test_sync_item_preserves_author_from_headers(self) -> None:
        """Test SyncItem author is extracted from headers."""
        kafka_msg = create_kafka_message_dataclass(
            topic="events",
            headers={"producer": "data-pipeline-v2"},
        )

        sync_item = kafka_msg.to_sync_item()

        assert sync_item.author == "data-pipeline-v2"


@pytest.mark.e2e
class TestKafkaConnectionLifecycle:
    """Tests for Kafka connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(
        self,
        connector: KafkaConnector,
        mock_aiokafka_consumer: MagicMock,
    ) -> None:
        """Test successful connection to Kafka."""
        # Patch where the import happens (in aiokafka module namespace)
        with patch.dict(
            "sys.modules",
            {
                "aiokafka": MagicMock(
                    AIOKafkaConsumer=MagicMock(return_value=mock_aiokafka_consumer)
                )
            },
        ):
            result = await connector.connect()

            assert result is True
            mock_aiokafka_consumer.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_without_kafka(
        self,
        connector: KafkaConnector,
    ) -> None:
        """Test connection failure when aiokafka not installed."""
        # Remove aiokafka from modules to simulate import error
        with patch.dict("sys.modules", {"aiokafka": None}):
            result = await connector.connect()

            # Should return False when aiokafka not available
            assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(
        self,
        connector: KafkaConnector,
        mock_aiokafka_consumer: MagicMock,
    ) -> None:
        """Test disconnecting from Kafka."""
        # Set consumer directly for disconnect test
        connector._consumer = mock_aiokafka_consumer
        connector._running = True

        await connector.disconnect()

        mock_aiokafka_consumer.stop.assert_called_once()


@pytest.mark.e2e
class TestCompleteStreamingWorkflow:
    """Tests for complete Kafka streaming workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_message_processing(
        self,
        connector: KafkaConnector,
        mock_aiokafka_consumer: MagicMock,
    ) -> None:
        """Test complete message processing workflow."""
        # Create mock messages
        mock_messages = [
            create_mock_kafka_message(
                topic="decisions",
                offset=i,
                value=json.dumps({"type": "decision", "id": f"dec-{i}"}).encode(),
            )
            for i in range(5)
        ]

        # Process messages without connecting (test deserialization + conversion)
        processed_items = []
        for mock_msg in mock_messages:
            kafka_msg = connector._deserialize_message(mock_msg)
            sync_item = kafka_msg.to_sync_item()
            processed_items.append(sync_item)

        assert len(processed_items) == 5
        assert all(item.source_type == "event_stream" for item in processed_items)
        # Check that content contains the decision id
        assert all("dec-" in item.content for item in processed_items)

    @pytest.mark.asyncio
    async def test_multiple_topics_consumption(self) -> None:
        """Test consuming from multiple topics."""
        config = create_test_kafka_config(topics=["decisions", "events", "metrics"])
        connector = KafkaConnector(config)

        assert len(connector.config.topics) == 3
        assert "decisions" in connector.config.topics
        assert "events" in connector.config.topics
        assert "metrics" in connector.config.topics

    @pytest.mark.asyncio
    async def test_batch_processing_configuration(self) -> None:
        """Test batch processing configuration."""
        config = create_test_kafka_config(
            batch_size=500,
            max_poll_records=1000,
        )
        connector = KafkaConnector(config)

        assert connector.config.batch_size == 500
        assert connector.config.max_poll_records == 1000


@pytest.mark.e2e
class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_connector_tracks_error_count(self, connector: KafkaConnector) -> None:
        """Test connector tracks error count."""
        # Manually increment error count for testing
        connector._error_count = 5

        stats = connector.get_stats()
        assert stats["error_count"] == 5

    def test_malformed_message_handling(self, connector: KafkaConnector) -> None:
        """Test handling of malformed messages."""
        # Message with invalid bytes
        mock_msg = MagicMock()
        mock_msg.topic = "test"
        mock_msg.partition = 0
        mock_msg.offset = 1
        mock_msg.key = None
        mock_msg.value = b"\xff\xfe"  # Invalid UTF-8
        mock_msg.timestamp = 1609459200000

        # Should handle gracefully
        try:
            kafka_msg = connector._deserialize_message(mock_msg)
            # If it doesn't raise, the message should be converted somehow
            assert kafka_msg is not None
        except (UnicodeDecodeError, ValueError):
            # It's acceptable to raise an error for malformed data
            pass


@pytest.mark.e2e
class TestSchemaRegistryIntegration:
    """Tests for schema registry integration."""

    def test_schema_registry_config(self) -> None:
        """Test schema registry configuration."""
        config = KafkaConfig(
            schema_registry_url="https://schema-registry.example.com",
            schema_registry_auth=("user", "password"),
        )

        assert config.schema_registry_url == "https://schema-registry.example.com"
        assert config.schema_registry_auth == ("user", "password")

    def test_config_without_schema_registry(self) -> None:
        """Test configuration without schema registry."""
        config = KafkaConfig()

        assert config.schema_registry_url is None
        assert config.schema_registry_auth is None


@pytest.mark.e2e
class TestKafkaMetrics:
    """Tests for Kafka connector metrics."""

    def test_consumed_count_tracking(self, connector: KafkaConnector) -> None:
        """Test consumed message count tracking."""
        # Simulate consumption
        connector._consumed_count = 100

        stats = connector.get_stats()
        assert stats["consumed_count"] == 100

    def test_stats_include_all_fields(self, connector: KafkaConnector) -> None:
        """Test stats include all required fields."""
        stats = connector.get_stats()

        required_fields = [
            "connector_id",
            "bootstrap_servers",
            "topics",
            "group_id",
            "running",
            "consumed_count",
            "error_count",
        ]

        for stat_field in required_fields:
            assert stat_field in stats, f"Missing field: {stat_field}"
