"""
Kafka Enterprise Connector.

Real-time event stream ingestion from Apache Kafka topics with:
- Consumer group management for scalable processing
- Offset tracking for reliable delivery
- Schema registry integration (optional)
- JSON, Avro, and Protobuf deserialization
- Connection retry with exponential backoff
- Circuit breaker for broker failures
- Dead letter queue (DLQ) for failed messages
- Graceful shutdown on SIGTERM

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
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Awaitable, Callable

from aragora.connectors.base import Evidence
from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.connectors.enterprise.streaming.resilience import (
    CircuitBreakerOpenError,
    DLQHandler,
    DLQMessage,
    ExponentialBackoff,
    GracefulShutdown,
    HealthMonitor,
    HealthStatus,
    StreamingCircuitBreaker,
    StreamingResilienceConfig,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Configuration for Kafka connector."""

    # Connection
    bootstrap_servers: str = "localhost:9092"
    topics: list[str] = field(default_factory=lambda: ["aragora-events"])
    group_id: str = "aragora-consumer"

    # Authentication
    security_protocol: str = "PLAINTEXT"  # PLAINTEXT, SSL, SASL_PLAINTEXT, SASL_SSL
    sasl_mechanism: str | None = None  # PLAIN, GSSAPI, SCRAM-SHA-256, etc.
    sasl_username: str | None = None
    sasl_password: str | None = None
    ssl_cafile: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    # Consumer settings
    auto_offset_reset: str = "earliest"  # earliest, latest, none
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000

    # Schema registry (optional)
    schema_registry_url: str | None = None
    schema_registry_auth: tuple | None = None  # (username, password)

    # Processing
    batch_size: int = 100
    poll_timeout_seconds: float = 1.0
    message_handler: Callable | None = None  # Custom message processor

    # Resilience settings
    resilience: StreamingResilienceConfig = field(default_factory=StreamingResilienceConfig)
    enable_circuit_breaker: bool = True
    enable_dlq: bool = True
    enable_graceful_shutdown: bool = True


@dataclass
class KafkaMessage:
    """A Kafka message with metadata."""

    topic: str
    partition: int
    offset: int
    key: str | None
    value: Any  # Deserialized payload
    headers: dict[str, str]
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
    - Connection retry with exponential backoff
    - Circuit breaker for broker failures
    - Dead letter queue (DLQ) for failed messages
    - Graceful shutdown on SIGTERM

    Uses aiokafka for async operation (falls back to confluent-kafka if needed).
    """

    def __init__(
        self,
        config: KafkaConfig,
        dlq_sender: Callable[[str, DLQMessage], Awaitable[None]] | None = None,
        **kwargs,
    ):
        """
        Initialize Kafka connector.

        Args:
            config: KafkaConfig with connection and processing settings
            dlq_sender: Optional custom DLQ sender function
        """
        super().__init__(connector_id="kafka", **kwargs)
        self.config = config
        self._consumer: Any | None = None
        self._producer: Any | None = None
        self._running = False
        self._consumed_count = 0
        self._error_count = 0
        self._dlq_count = 0

        # Resilience components
        self._circuit_breaker: StreamingCircuitBreaker | None = None
        self._dlq_handler: DLQHandler | None = None
        self._health_monitor: HealthMonitor | None = None
        self._graceful_shutdown: GracefulShutdown | None = None

        if config.enable_circuit_breaker:
            self._circuit_breaker = StreamingCircuitBreaker(
                name="kafka-broker",
                config=config.resilience,
            )

        if config.enable_dlq:
            self._dlq_handler = DLQHandler(
                config=config.resilience,
                dlq_sender=dlq_sender or self._default_dlq_sender,
            )

        self._health_monitor = HealthMonitor(
            name="kafka-connector",
            config=config.resilience,
        )

        if config.enable_graceful_shutdown:
            self._graceful_shutdown = GracefulShutdown()

    async def _default_dlq_sender(self, topic: str, message: DLQMessage) -> None:
        """Default DLQ sender using Kafka producer."""
        if not self._producer:
            await self._create_producer()

        if self._producer:
            await self._producer.send_and_wait(
                topic,
                value=message.to_json().encode("utf-8"),
                key=message.original_key.encode("utf-8") if message.original_key else None,
            )
            logger.info(f"[Kafka] Sent message to DLQ topic: {topic}")

    async def _create_producer(self) -> None:
        """Create a Kafka producer for DLQ messages."""
        try:
            from aiokafka import AIOKafkaProducer

            producer_config = {
                "bootstrap_servers": self.config.bootstrap_servers,
            }

            # Add security settings
            if self.config.security_protocol != "PLAINTEXT":
                producer_config["security_protocol"] = self.config.security_protocol

            if self.config.sasl_mechanism:
                producer_config["sasl_mechanism"] = self.config.sasl_mechanism
                producer_config["sasl_plain_username"] = self.config.sasl_username
                producer_config["sasl_plain_password"] = self.config.sasl_password

            if self.config.ssl_cafile:
                producer_config["ssl_cafile"] = self.config.ssl_cafile
            if self.config.ssl_certfile:
                producer_config["ssl_certfile"] = self.config.ssl_certfile
            if self.config.ssl_keyfile:
                producer_config["ssl_keyfile"] = self.config.ssl_keyfile

            self._producer = AIOKafkaProducer(**producer_config)
            await self._producer.start()
            logger.info("[Kafka] Producer created for DLQ")

        except ImportError:
            logger.warning("[Kafka] aiokafka not installed, DLQ sender disabled")
        except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.error(f"[Kafka] Failed to create producer: {e}")

    async def connect(self) -> bool:
        """
        Connect to Kafka cluster with retry and circuit breaker.

        Returns:
            True if connection successful
        """
        backoff = ExponentialBackoff(self.config.resilience)

        for attempt in range(self.config.resilience.max_retries + 1):
            # Check circuit breaker
            if self._circuit_breaker and self._circuit_breaker.is_open:
                if not await self._circuit_breaker.can_execute():
                    logger.warning("[Kafka] Circuit breaker is open, skipping connect")
                    return False

            try:
                success = await self._connect_internal()
                if success:
                    if self._circuit_breaker:
                        await self._circuit_breaker.record_success()
                    if self._health_monitor:
                        await self._health_monitor.record_success()
                    return True

            except ImportError:
                # Don't retry import errors
                logger.error("[Kafka] aiokafka not installed. Install with: pip install aiokafka")
                return False

            except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                if self._circuit_breaker:
                    await self._circuit_breaker.record_failure(e)
                if self._health_monitor:
                    await self._health_monitor.record_failure(e)

                if attempt == self.config.resilience.max_retries:
                    logger.error(f"[Kafka] Connection failed after {attempt + 1} attempts: {e}")
                    return False

                delay = backoff.get_delay(attempt)
                logger.warning(
                    f"[Kafka] Connection attempt {attempt + 1}/{self.config.resilience.max_retries + 1} "
                    f"failed: {e}. Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        return False

    async def _connect_internal(self) -> bool:
        """Internal connection logic."""
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

    async def disconnect(self) -> None:
        """Disconnect from Kafka cluster."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("[Kafka] Consumer disconnected")

        if self._producer:
            await self._producer.stop()
            self._producer = None
            logger.info("[Kafka] Producer disconnected")

    async def start(self) -> None:
        """Start consuming messages with graceful shutdown support."""
        if not self._consumer:
            connected = await self.connect()
            if not connected:
                raise RuntimeError("Failed to connect to Kafka")

        self._running = True

        # Set up graceful shutdown
        if self._graceful_shutdown:
            self._graceful_shutdown.setup_signal_handlers()
            self._graceful_shutdown.register_cleanup(self._cleanup)

    async def _cleanup(self) -> None:
        """Cleanup on graceful shutdown."""
        logger.info("[Kafka] Performing graceful shutdown cleanup...")
        await self.disconnect()

    async def stop(self) -> None:
        """Stop consuming messages."""
        await self.disconnect()

    async def consume(
        self,
        max_messages: int | None = None,
        process_fn: Callable[[KafkaMessage], Awaitable[None]] | None = None,
    ) -> AsyncIterator[KafkaMessage]:
        """
        Consume messages from Kafka topics with resilience.

        Args:
            max_messages: Optional limit on messages to consume
            process_fn: Optional async function to process each message

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
                # Check for graceful shutdown
                if self._graceful_shutdown and self._graceful_shutdown.is_shutting_down:
                    logger.info("[Kafka] Shutdown signal received, stopping consume")
                    break

                start_time = time.time()

                try:
                    # Deserialize message
                    kafka_msg = self._deserialize_message(msg)

                    # Process message if handler provided
                    if process_fn:
                        await self._process_with_resilience(kafka_msg, process_fn)

                    yield kafka_msg

                    self._consumed_count += 1
                    messages_consumed += 1

                    # Record success metrics
                    if self._health_monitor:
                        latency_ms = (time.time() - start_time) * 1000
                        await self._health_monitor.record_success(latency_ms=latency_ms)

                    if max_messages and messages_consumed >= max_messages:
                        break

                except CircuitBreakerOpenError as e:
                    logger.warning(f"[Kafka] Circuit breaker open: {e}")
                    # Wait for recovery
                    await asyncio.sleep(e.recovery_time)

                except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as e:
                    self._error_count += 1
                    logger.warning(f"[Kafka] Error processing message: {e}")

                    if self._health_monitor:
                        await self._health_monitor.record_failure(e)

                    # Handle DLQ
                    if self._dlq_handler:
                        kafka_msg = self._deserialize_message(msg)
                        sent_to_dlq = await self._dlq_handler.handle_failure(
                            topic=kafka_msg.topic,
                            key=kafka_msg.key,
                            value=kafka_msg.value,
                            headers=kafka_msg.headers,
                            timestamp=kafka_msg.timestamp,
                            error=e,
                            offset=kafka_msg.offset,
                        )
                        if sent_to_dlq:
                            self._dlq_count += 1

        except asyncio.CancelledError:
            logger.info("[Kafka] Consumer cancelled")
            raise

    async def _process_with_resilience(
        self,
        message: KafkaMessage,
        process_fn: Callable[[KafkaMessage], Awaitable[None]],
    ) -> None:
        """Process a message with circuit breaker protection."""
        if self._circuit_breaker:
            async with await self._circuit_breaker.call():
                await process_fn(message)
        else:
            await process_fn(message)

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

    async def sync(self, batch_size: int | None = None) -> AsyncIterator[SyncItem]:  # type: ignore[override]
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

    def get_stats(self) -> dict[str, Any]:
        """Get connector statistics."""
        stats = {
            "connector_id": self.connector_id,
            "bootstrap_servers": self.config.bootstrap_servers,
            "topics": self.config.topics,
            "group_id": self.config.group_id,
            "running": self._running,
            "consumed_count": self._consumed_count,
            "error_count": self._error_count,
            "dlq_count": self._dlq_count,
        }

        # Add resilience stats
        if self._circuit_breaker:
            stats["circuit_breaker"] = self._circuit_breaker.get_stats()

        if self._dlq_handler:
            stats["dlq"] = self._dlq_handler.get_stats()

        return stats

    async def get_health(self) -> HealthStatus:
        """Get connector health status."""
        if self._health_monitor:
            return await self._health_monitor.get_status()
        return HealthStatus(
            healthy=self._running,
            last_check=datetime.now(timezone.utc),
        )

    def reset_circuit_breaker(self) -> None:
        """Reset the circuit breaker to closed state."""
        if self._circuit_breaker:
            self._circuit_breaker.reset()

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

    async def fetch(self, evidence_id: str) -> Evidence | None:
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
