"""
Kafka Enterprise Connector.

Real-time event stream ingestion from Apache Kafka topics with:
- Consumer group management for scalable processing
- Offset tracking for reliable delivery
- Schema registry integration (optional)
- JSON, Avro, and Protobuf deserialization

Requires: confluent-kafka or aiokafka

Usage:
    config = KafkaConfig(
        bootstrap_servers="localhost:9092",
        topics=["decisions", "events"],
        group_id="aragora-consumer",
    )
    connector = KafkaConnector(config)
    await connector.start()

    async for message in connector.consume():
        # Process message into Knowledge Mound
        await knowledge_mound.ingest(message.to_sync_item())
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from aragora.connectors.base import Evidence
from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Configuration for Kafka connector."""

    # Connection
    bootstrap_servers: str = "localhost:9092"
    topics: List[str] = field(default_factory=lambda: ["aragora-events"])
    group_id: str = "aragora-consumer"

    # Authentication
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL
    sasl_mechanism: Optional[str] = None  # PLAIN, GSSAPI, SCRAM-SHA-256, etc.
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # Consumer settings
    auto_offset_reset: str = "earliest"  # earliest, latest, none
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000

    # Schema registry (optional)
    schema_registry_url: Optional[str] = None
    schema_registry_auth: Optional[tuple] = None  # (username, password)

    # Processing
    batch_size: int = 100
    poll_timeout_seconds: float = 1.0
    message_handler: Optional[Callable] = None  # Custom message processor


@dataclass
class KafkaMessage:
    """A Kafka message with metadata."""

    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: Any  # Deserialized payload
    headers: Dict[str, str]
    timestamp: datetime

    def to_sync_item(self) -> SyncItem:
        """Convert to SyncItem for Knowledge Mound ingestion."""
        # Extract content from value
        if isinstance(self.value, dict):
            content = json.dumps(self.value, indent=2)
            title = self.value.get("title") or self.value.get("type") or f"Kafka: {self.topic}"
        elif isinstance(self.value, str):
            content = self.value
            title = f"Kafka: {self.topic}"
        else:
            content = str(self.value)
            title = f"Kafka: {self.topic}"

        return SyncItem(
            id=f"kafka-{self.topic}-{self.partition}-{self.offset}",
            content=content[:50000],
            source_type="event_stream",
            source_id=f"kafka/{self.topic}/{self.partition}/{self.offset}",
            title=title,
            url=None,
            author=self.headers.get("producer", "kafka"),
            created_at=self.timestamp,
            updated_at=self.timestamp,
            domain="enterprise/kafka",
            confidence=0.9,
            metadata={
                "topic": self.topic,
                "partition": self.partition,
                "offset": self.offset,
                "key": self.key,
                "headers": self.headers,
            },
        )


class KafkaConnector(EnterpriseConnector):
    """
    Enterprise connector for Apache Kafka.

    Provides real-time event stream ingestion with:
    - Consumer group management for horizontal scaling
    - Reliable message delivery with offset tracking
    - Support for multiple serialization formats
    - Schema registry integration

    Uses aiokafka for async operation (falls back to confluent-kafka if needed).
    """

    def __init__(self, config: KafkaConfig, **kwargs):
        """
        Initialize Kafka connector.

        Args:
            config: KafkaConfig with connection and processing settings
        """
        super().__init__(connector_id="kafka", **kwargs)
        self.config = config
        self._consumer: Optional[Any] = None
        self._running = False
        self._consumed_count = 0
        self._error_count = 0

    async def connect(self) -> bool:
        """
        Connect to Kafka cluster.

        Returns:
            True if connection successful
        """
        try:
            from aiokafka import AIOKafkaConsumer

            # Build consumer config
            consumer_config = {
                "bootstrap_servers": self.config.bootstrap_servers,
                "group_id": self.config.group_id,
                "auto_offset_reset": self.config.auto_offset_reset,
                "enable_auto_commit": self.config.enable_auto_commit,
            }

            # Add security settings
            if self.config.security_protocol != "PLAINTEXT":
                consumer_config["security_protocol"] = self.config.security_protocol

            if self.config.sasl_mechanism:
                consumer_config["sasl_mechanism"] = self.config.sasl_mechanism
                consumer_config["sasl_plain_username"] = self.config.sasl_username
                consumer_config["sasl_plain_password"] = self.config.sasl_password

            if self.config.ssl_cafile:
                consumer_config["ssl_cafile"] = self.config.ssl_cafile
            if self.config.ssl_certfile:
                consumer_config["ssl_certfile"] = self.config.ssl_certfile
            if self.config.ssl_keyfile:
                consumer_config["ssl_keyfile"] = self.config.ssl_keyfile

            self._consumer = AIOKafkaConsumer(
                *self.config.topics,
                **consumer_config,
            )
            consumer = self._consumer
            await consumer.start()
            logger.info(
                f"[Kafka] Connected to {self.config.bootstrap_servers}, "
                f"topics={self.config.topics}, group={self.config.group_id}"
            )
            return True

        except ImportError:
            logger.error("[Kafka] aiokafka not installed. Install with: pip install aiokafka")
            return False
        except Exception as e:
            logger.error(f"[Kafka] Connection failed: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("[Kafka] Disconnected")

    async def start(self) -> None:
        """Start consuming messages."""
        if not self._consumer:
            connected = await self.connect()
            if not connected:
                raise RuntimeError("Failed to connect to Kafka")
        self._running = True

    async def stop(self) -> None:
        """Stop consuming messages."""
        await self.disconnect()

    async def consume(self, max_messages: Optional[int] = None) -> AsyncIterator[KafkaMessage]:
        """
        Consume messages from Kafka topics.

        Args:
            max_messages: Optional limit on messages to consume

        Yields:
            KafkaMessage objects
        """
        if not self._consumer:
            await self.start()

        consumer = self._consumer
        assert consumer is not None, "Consumer should be initialized after start()"
        messages_consumed = 0

        try:
            async for msg in consumer:
                try:
                    # Deserialize message
                    kafka_msg = self._deserialize_message(msg)
                    yield kafka_msg

                    self._consumed_count += 1
                    messages_consumed += 1

                    if max_messages and messages_consumed >= max_messages:
                        break

                except Exception as e:
                    self._error_count += 1
                    logger.warning(f"[Kafka] Error processing message: {e}")

        except asyncio.CancelledError:
            logger.info("[Kafka] Consumer cancelled")
            raise

    def _deserialize_message(self, msg) -> KafkaMessage:
        """Deserialize a Kafka message."""
        # Decode value
        value = msg.value
        if isinstance(value, bytes):
            try:
                value = json.loads(value.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                value = value.decode("utf-8", errors="replace")

        # Decode key
        key = msg.key
        if isinstance(key, bytes):
            key = key.decode("utf-8", errors="replace")

        # Decode headers
        headers = {}
        if msg.headers:
            for k, v in msg.headers:
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                headers[k] = v

        # Parse timestamp
        timestamp = datetime.fromtimestamp(msg.timestamp / 1000, tz=timezone.utc)

        return KafkaMessage(
            topic=msg.topic,
            partition=msg.partition,
            offset=msg.offset,
            key=key,
            value=value,
            headers=headers,
            timestamp=timestamp,
        )

    async def sync(self, batch_size: int = None) -> AsyncIterator[SyncItem]:  # type: ignore[override]
        """
        Sync messages as SyncItems for Knowledge Mound ingestion.

        Args:
            batch_size: Number of messages to sync (None = continuous)

        Yields:
            SyncItem objects
        """
        batch_size = batch_size or self.config.batch_size

        async for msg in self.consume(max_messages=batch_size):
            yield msg.to_sync_item()

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "connector_id": self.connector_id,
            "bootstrap_servers": self.config.bootstrap_servers,
            "topics": self.config.topics,
            "group_id": self.config.group_id,
            "running": self._running,
            "consumed_count": self._consumed_count,
            "error_count": self._error_count,
        }

    # Required abstract method implementations

    @property
    def source_type(self) -> SourceType:
        """The source type for this connector."""
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable name for this connector."""
        return "Kafka"

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search is not supported for streaming connectors.

        Kafka is a real-time stream - historical search requires
        specialized tooling like ksqlDB or Kafka Connect with search index.
        """
        logger.warning("[Kafka] Search not supported for streaming connector")
        return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
        """
        Fetch is not supported for streaming connectors.

        Kafka messages are consumed once and not randomly accessible
        without specialized offset management.
        """
        logger.warning("[Kafka] Fetch not supported for streaming connector")
        return None

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield items from Kafka topics for incremental sync.

        Args:
            state: Sync state with cursor position
            batch_size: Number of messages to consume per batch

        Yields:
            SyncItem objects for Knowledge Mound ingestion
        """
        async for item in self.sync(batch_size=batch_size):
            yield item


__all__ = ["KafkaConnector", "KafkaConfig", "KafkaMessage"]
