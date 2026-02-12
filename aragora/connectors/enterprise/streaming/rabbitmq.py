"""
RabbitMQ Enterprise Connector.

Message queue ingestion from RabbitMQ with:
- Exchange and queue binding management
- Message acknowledgment for reliable delivery
- Dead letter queue handling
- JSON and binary message deserialization
- Connection retry with exponential backoff
- Circuit breaker for broker failures
- Dead letter queue (DLQ) for failed messages
- Graceful shutdown on SIGTERM

Requires: aio-pika

Usage:
    config = RabbitMQConfig(
        url=os.environ["RABBITMQ_URL"],  # Required, no default
        queue="decisions",
        exchange="aragora",
    )
    connector = RabbitMQConnector(config)
    await connector.start()

    async for message in connector.consume():
        # Process message into Knowledge Mound
        await knowledge_mound.ingest(message.to_sync_item())
        await message.ack()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import AsyncIterator, Awaitable, Callable

from aragora.connectors.base import Evidence

# Type hint for aio_pika channel (optional import)
if TYPE_CHECKING:
    from aio_pika import Channel, Queue, RobustConnection
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
class RabbitMQConfig:
    """Configuration for RabbitMQ connector.

    Note: The url parameter has no default value to prevent accidental use of
    insecure credentials in production. Always explicitly configure the URL.
    """

    # Connection - NO DEFAULT to prevent insecure production deployments
    url: str  # Required: e.g., "amqp://user:password@host:5672/vhost"
    queue: str = "aragora-events"
    exchange: str = ""  # Default exchange if empty
    exchange_type: str = "direct"  # direct, fanout, topic, headers
    routing_key: str = ""

    # Queue settings
    durable: bool = True
    auto_delete: bool = False
    exclusive: bool = False
    prefetch_count: int = 10  # QoS prefetch

    # Dead letter queue
    dead_letter_exchange: str | None = None
    dead_letter_routing_key: str | None = None
    message_ttl: int | None = None  # milliseconds

    # SSL/TLS
    ssl: bool = False
    ssl_cafile: str | None = None
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    # Processing
    batch_size: int = 100
    auto_ack: bool = False  # Manual ack for reliability
    requeue_on_error: bool = True
    message_handler: Callable | None = None

    # Resilience settings
    resilience: StreamingResilienceConfig = field(default_factory=StreamingResilienceConfig)
    enable_circuit_breaker: bool = True
    enable_dlq: bool = True
    enable_graceful_shutdown: bool = True


@dataclass
class RabbitMQMessage:
    """A RabbitMQ message with metadata."""

    queue: str
    routing_key: str
    delivery_tag: int
    body: Any  # Deserialized payload
    headers: dict[str, str]
    timestamp: datetime
    message_id: str | None = None
    correlation_id: str | None = None
    reply_to: str | None = None
    expiration: str | None = None
    priority: int | None = None

    # Internal reference for ack/nack
    _channel: Any = field(default=None, repr=False)

    async def ack(self) -> None:
        """Acknowledge the message."""
        if self._channel:
            await self._channel.basic_ack(self.delivery_tag)

    async def nack(self, requeue: bool = True) -> None:
        """Negative acknowledge the message."""
        if self._channel:
            await self._channel.basic_nack(self.delivery_tag, requeue=requeue)

    async def reject(self, requeue: bool = False) -> None:
        """Reject the message."""
        if self._channel:
            await self._channel.basic_reject(self.delivery_tag, requeue=requeue)

    def to_sync_item(self) -> SyncItem:
        """Convert to SyncItem for Knowledge Mound ingestion."""
        # Extract content from body
        if isinstance(self.body, dict):
            content = json.dumps(self.body, indent=2)
            title = self.body.get("title") or self.body.get("type") or f"RabbitMQ: {self.queue}"
        elif isinstance(self.body, str):
            content = self.body
            title = f"RabbitMQ: {self.queue}"
        else:
            content = str(self.body)
            title = f"RabbitMQ: {self.queue}"

        return SyncItem(
            id=f"rabbitmq-{self.queue}-{self.delivery_tag}",
            content=content[:50000],
            source_type="message_queue",
            source_id=f"rabbitmq/{self.queue}/{self.delivery_tag}",
            title=title,
            url=None,
            author=self.headers.get("producer", "rabbitmq"),
            created_at=self.timestamp,
            updated_at=self.timestamp,
            domain="enterprise/rabbitmq",
            confidence=0.9,
            metadata={
                "queue": self.queue,
                "routing_key": self.routing_key,
                "delivery_tag": self.delivery_tag,
                "message_id": self.message_id,
                "correlation_id": self.correlation_id,
                "headers": self.headers,
                "priority": self.priority,
            },
        )


class RabbitMQConnector(EnterpriseConnector):
    """
    Enterprise connector for RabbitMQ.

    Provides message queue ingestion with:
    - Exchange and queue binding management
    - Reliable message delivery with manual acknowledgment
    - Dead letter queue support for failed messages
    - Support for multiple serialization formats
    - Connection retry with exponential backoff
    - Circuit breaker for broker failures
    - Graceful shutdown on SIGTERM

    Uses aio-pika for async operation.
    """

    def __init__(
        self,
        config: RabbitMQConfig,
        dlq_sender: Callable[[str, DLQMessage], Awaitable[None]] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize RabbitMQ connector.

        Args:
            config: RabbitMQConfig with connection and processing settings
            dlq_sender: Optional custom DLQ sender function
        """
        # RabbitMQ uses streaming-specific circuit breaker; disable base breaker.
        kwargs.pop("enable_circuit_breaker", None)
        super().__init__(connector_id="rabbitmq", enable_circuit_breaker=False, **kwargs)
        self.config = config
        self._connection: RobustConnection | None = None
        self._channel: Channel | None = None
        self._queue: Queue | None = None
        self._running = False
        self._consumed_count = 0
        self._error_count = 0
        self._acked_count = 0
        self._nacked_count = 0
        self._dlq_count = 0

        # Resilience components (use different names to avoid base class conflict)
        self._streaming_circuit_breaker: StreamingCircuitBreaker | None = None
        self._dlq_handler: DLQHandler | None = None
        self._health_monitor: HealthMonitor | None = None
        self._graceful_shutdown: GracefulShutdown | None = None

        self._enable_circuit_breaker = config.enable_circuit_breaker
        if config.enable_circuit_breaker:
            self._streaming_circuit_breaker = StreamingCircuitBreaker(
                name="rabbitmq-broker",
                config=config.resilience,
            )
            # Alias for compatibility with base connector/tests.
            self._circuit_breaker = self._streaming_circuit_breaker  # type: ignore[assignment]
        else:
            self._circuit_breaker = None

        if config.enable_dlq:
            self._dlq_handler = DLQHandler(
                config=config.resilience,
                dlq_sender=dlq_sender or self._default_dlq_sender,
            )

        self._health_monitor = HealthMonitor(
            name="rabbitmq-connector",
            config=config.resilience,
        )

        if config.enable_graceful_shutdown:
            self._graceful_shutdown = GracefulShutdown()

    async def _default_dlq_sender(self, queue_name: str, message: DLQMessage) -> None:
        """Default DLQ sender using RabbitMQ channel."""
        if not self._channel:
            logger.warning("[RabbitMQ] No channel available for DLQ sender")
            return

        try:
            import aio_pika

            # Build DLQ queue name
            dlq_queue = queue_name

            # Declare DLQ queue if it doesn't exist
            await self._channel.declare_queue(
                dlq_queue,
                durable=self.config.durable,
            )

            # Create message
            msg = aio_pika.Message(
                body=message.to_json().encode("utf-8"),
                headers={
                    "x-original-topic": message.original_topic,
                    "x-error-type": message.error_type,
                    "x-error-message": message.error_message[:500],  # Limit error message
                    "x-retry-count": str(message.retry_count),
                },
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                if self.config.durable
                else aio_pika.DeliveryMode.NOT_PERSISTENT,
            )

            # Publish to DLQ
            await self._channel.default_exchange.publish(
                msg,
                routing_key=dlq_queue,
            )
            logger.info(f"[RabbitMQ] Sent message to DLQ: {dlq_queue}")

        except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.error(f"[RabbitMQ] Failed to send to DLQ: {e}")
            raise

    async def connect(self) -> bool:
        """
        Connect to RabbitMQ server with retry and circuit breaker.

        Returns:
            True if connection successful
        """
        backoff = ExponentialBackoff(self.config.resilience)

        for attempt in range(self.config.resilience.max_retries + 1):
            # Check circuit breaker
            if self._streaming_circuit_breaker and self._streaming_circuit_breaker.is_open:
                if not await self._streaming_circuit_breaker.can_execute():
                    logger.warning("[RabbitMQ] Circuit breaker is open, skipping connect")
                    return False

            try:
                success = await self._connect_internal()
                if success:
                    if self._streaming_circuit_breaker:
                        await self._streaming_circuit_breaker.record_success()
                    if self._health_monitor:
                        await self._health_monitor.record_success()
                    return True

            except ImportError:
                # Don't retry import errors
                logger.error(
                    "[RabbitMQ] aio-pika not installed. Install with: pip install aio-pika"
                )
                return False

            except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError) as e:
                if self._streaming_circuit_breaker:
                    await self._streaming_circuit_breaker.record_failure(e)
                if self._health_monitor:
                    await self._health_monitor.record_failure(e)

                if attempt == self.config.resilience.max_retries:
                    logger.error(f"[RabbitMQ] Connection failed after {attempt + 1} attempts: {e}")
                    return False

                delay = backoff.get_delay(attempt)
                logger.warning(
                    f"[RabbitMQ] Connection attempt {attempt + 1}/{self.config.resilience.max_retries + 1} "
                    f"failed: {e}. Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        return False

    async def _connect_internal(self) -> bool:
        """Internal connection logic."""
        import aio_pika

        # Build connection URL with SSL if needed
        url = self.config.url
        if self.config.ssl:
            import ssl as ssl_module

            ssl_context = ssl_module.create_default_context()
            if self.config.ssl_cafile:
                ssl_context.load_verify_locations(self.config.ssl_cafile)
            if self.config.ssl_certfile and self.config.ssl_keyfile:
                ssl_context.load_cert_chain(
                    self.config.ssl_certfile,
                    self.config.ssl_keyfile,
                )
            self._connection = await aio_pika.connect_robust(url, ssl_context=ssl_context)
        else:
            self._connection = await aio_pika.connect_robust(url)

        # Create channel with QoS - connection is guaranteed to be set at this point
        if self._connection is None:
            raise RuntimeError("Failed to establish RabbitMQ connection")
        self._channel = await self._connection.channel()
        if self._channel is None:
            raise RuntimeError("Failed to create RabbitMQ channel")
        await self._channel.set_qos(prefetch_count=self.config.prefetch_count)

        # Declare exchange if specified
        if self.config.exchange:
            exchange = await self._channel.declare_exchange(
                self.config.exchange,
                type=self.config.exchange_type,
                durable=self.config.durable,
            )
        else:
            exchange = self._channel.default_exchange

        # Build queue arguments
        queue_args: dict[str, Any] = {}
        if self.config.dead_letter_exchange:
            queue_args["x-dead-letter-exchange"] = self.config.dead_letter_exchange
        if self.config.dead_letter_routing_key:
            queue_args["x-dead-letter-routing-key"] = self.config.dead_letter_routing_key
        if self.config.message_ttl:
            queue_args["x-message-ttl"] = self.config.message_ttl

        # Declare queue
        self._queue = await self._channel.declare_queue(
            self.config.queue,
            durable=self.config.durable,
            auto_delete=self.config.auto_delete,
            exclusive=self.config.exclusive,
            arguments=queue_args or None,
        )

        # Bind queue to exchange if specified
        if self.config.exchange:
            if self._queue is None:
                raise RuntimeError("Failed to declare RabbitMQ queue")
            await self._queue.bind(
                exchange,
                routing_key=self.config.routing_key or self.config.queue,
            )

        logger.info(
            f"[RabbitMQ] Connected to {self.config.url}, "
            f"queue={self.config.queue}, exchange={self.config.exchange or 'default'}"
        )
        return True

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ server."""
        self._running = False
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._channel = None
            self._queue = None
            logger.info("[RabbitMQ] Disconnected")

    async def start(self) -> None:
        """Start consuming messages with graceful shutdown support."""
        if not self._connection:
            connected = await self.connect()
            if not connected:
                raise RuntimeError("Failed to connect to RabbitMQ")
        self._running = True

        # Set up graceful shutdown
        if self._graceful_shutdown:
            self._graceful_shutdown.setup_signal_handlers()
            self._graceful_shutdown.register_cleanup(self._cleanup)

    async def _cleanup(self) -> None:
        """Cleanup on graceful shutdown."""
        logger.info("[RabbitMQ] Performing graceful shutdown cleanup...")
        await self.disconnect()

    async def stop(self) -> None:
        """Stop consuming messages."""
        await self.disconnect()

    async def consume(
        self,
        max_messages: int | None = None,
        process_fn: Callable[[RabbitMQMessage], Awaitable[None]] | None = None,
    ) -> AsyncIterator[RabbitMQMessage]:
        """
        Consume messages from RabbitMQ queue with resilience.

        Args:
            max_messages: Optional limit on messages to consume
            process_fn: Optional async function to process each message

        Yields:
            RabbitMQMessage objects
        """
        if not self._queue:
            await self.start()

        messages_consumed = 0
        if self._queue is None:
            raise RuntimeError("RabbitMQ queue not initialized - call start() first")

        try:
            async with self._queue.iterator() as queue_iter:
                async for message in queue_iter:
                    # Check for graceful shutdown
                    if self._graceful_shutdown and self._graceful_shutdown.is_shutting_down:
                        logger.info("[RabbitMQ] Shutdown signal received, stopping consume")
                        break

                    start_time = time.time()

                    try:
                        # Deserialize message
                        rabbitmq_msg = self._deserialize_message(message)

                        # Process message if handler provided
                        if process_fn:
                            await self._process_with_resilience(rabbitmq_msg, process_fn)

                        yield rabbitmq_msg

                        self._consumed_count += 1
                        messages_consumed += 1

                        # Record success metrics
                        if self._health_monitor:
                            latency_ms = (time.time() - start_time) * 1000
                            await self._health_monitor.record_success(latency_ms=latency_ms)

                        # Auto-ack if configured
                        if self.config.auto_ack:
                            await message.ack()
                            self._acked_count += 1

                        if max_messages and messages_consumed >= max_messages:
                            break

                    except CircuitBreakerOpenError as e:
                        logger.warning(f"[RabbitMQ] Circuit breaker open: {e}")
                        # Requeue and wait for recovery
                        await message.nack(requeue=True)
                        await asyncio.sleep(e.recovery_time)

                    except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as e:
                        self._error_count += 1
                        logger.warning(f"[RabbitMQ] Error processing message: {e}")

                        if self._health_monitor:
                            await self._health_monitor.record_failure(e)

                        # Handle DLQ
                        if self._dlq_handler:
                            rabbitmq_msg = self._deserialize_message(message)
                            sent_to_dlq = await self._dlq_handler.handle_failure(
                                topic=rabbitmq_msg.queue,
                                key=rabbitmq_msg.message_id,
                                value=rabbitmq_msg.body,
                                headers=rabbitmq_msg.headers,
                                timestamp=rabbitmq_msg.timestamp,
                                error=e,
                                delivery_tag=rabbitmq_msg.delivery_tag,
                            )
                            if sent_to_dlq:
                                self._dlq_count += 1
                                # Acknowledge the original message since it went to DLQ
                                await message.ack()
                                self._acked_count += 1
                            elif self.config.requeue_on_error:
                                await message.nack(requeue=True)
                            else:
                                await message.reject(requeue=False)
                                self._nacked_count += 1
                        elif self.config.requeue_on_error:
                            await message.nack(requeue=True)
                        else:
                            await message.reject(requeue=False)
                        self._nacked_count += 1

        except asyncio.CancelledError:
            logger.info("[RabbitMQ] Consumer cancelled")
            raise

    async def _process_with_resilience(
        self,
        message: RabbitMQMessage,
        process_fn: Callable[[RabbitMQMessage], Awaitable[None]],
    ) -> None:
        """Process a message with circuit breaker protection."""
        if self._streaming_circuit_breaker:
            async with await self._streaming_circuit_breaker.call():
                await process_fn(message)
        else:
            await process_fn(message)

    def _deserialize_message(self, message: Any) -> RabbitMQMessage:
        """Deserialize a RabbitMQ message."""
        # Decode body
        body = message.body
        if isinstance(body, bytes):
            try:
                body = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = body.decode("utf-8", errors="replace")

        # Extract headers
        headers = {}
        if message.headers:
            for k, v in message.headers.items():
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                headers[str(k)] = str(v) if v is not None else ""

        # Parse timestamp
        if message.timestamp:
            timestamp = message.timestamp
            if not isinstance(timestamp, datetime):
                timestamp = datetime.fromtimestamp(float(message.timestamp), tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        return RabbitMQMessage(
            queue=self.config.queue,
            routing_key=message.routing_key or "",
            delivery_tag=message.delivery_tag,
            body=body,
            headers=headers,
            timestamp=timestamp,
            message_id=message.message_id,
            correlation_id=message.correlation_id,
            reply_to=message.reply_to,
            expiration=message.expiration,
            priority=message.priority,
            _channel=message.channel,
        )

    async def sync(  # type: ignore[override]  # streaming connector uses batch_size instead of full_sync param
        self, batch_size: int | None = None
    ) -> AsyncIterator[SyncItem]:
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
            # Auto-ack after successful conversion
            if not self.config.auto_ack:
                await msg.ack()
                self._acked_count += 1

    async def publish(
        self,
        body: Any,
        routing_key: str | None = None,
        headers: dict[str, str] | None = None,
        message_id: str | None = None,
        correlation_id: str | None = None,
        reply_to: str | None = None,
        expiration: str | None = None,
        priority: int | None = None,
    ) -> bool:
        """
        Publish a message to RabbitMQ with resilience.

        Args:
            body: Message body (will be JSON serialized if dict)
            routing_key: Routing key (defaults to queue name)
            headers: Message headers
            message_id: Message ID
            correlation_id: Correlation ID for RPC
            reply_to: Reply queue name
            expiration: Message TTL
            priority: Message priority (0-9)

        Returns:
            True if published successfully
        """
        # Check circuit breaker
        if self._streaming_circuit_breaker:
            if not await self._streaming_circuit_breaker.can_execute():
                raise CircuitBreakerOpenError(
                    self._streaming_circuit_breaker.name,
                    self._streaming_circuit_breaker._time_until_recovery(),
                )

        try:
            import aio_pika

            if not self._channel:
                await self.connect()

            # Serialize body
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")

            # Create message
            message = aio_pika.Message(
                body=body,
                headers=headers,
                message_id=message_id,
                correlation_id=correlation_id,
                reply_to=reply_to,
                expiration=expiration,
                priority=priority,
                delivery_mode=(
                    aio_pika.DeliveryMode.PERSISTENT
                    if self.config.durable
                    else aio_pika.DeliveryMode.NOT_PERSISTENT
                ),
            )

            # Get exchange - channel is guaranteed to be set after connect
            if self._channel is None:
                raise RuntimeError("RabbitMQ channel not initialized - call start() first")
            if self.config.exchange:
                exchange = await self._channel.get_exchange(self.config.exchange)
            else:
                exchange = self._channel.default_exchange

            # Publish
            await exchange.publish(
                message,
                routing_key=routing_key or self.config.routing_key or self.config.queue,
            )

            if self._streaming_circuit_breaker:
                await self._streaming_circuit_breaker.record_success()
            if self._health_monitor:
                await self._health_monitor.record_success()

            logger.debug(f"[RabbitMQ] Published message to {routing_key or self.config.queue}")
            return True

        except (OSError, ConnectionError, TimeoutError, RuntimeError, ValueError, json.JSONDecodeError) as e:
            if self._streaming_circuit_breaker:
                await self._streaming_circuit_breaker.record_failure(e)
            if self._health_monitor:
                await self._health_monitor.record_failure(e)
            logger.error(f"[RabbitMQ] Failed to publish message: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get connector statistics."""
        stats = {
            "connector_id": self.connector_id,
            "url": self.config.url.split("@")[-1],  # Hide credentials
            "queue": self.config.queue,
            "exchange": self.config.exchange or "default",
            "running": self._running,
            "consumed_count": self._consumed_count,
            "acked_count": self._acked_count,
            "nacked_count": self._nacked_count,
            "error_count": self._error_count,
            "dlq_count": self._dlq_count,
        }

        # Add resilience stats
        if self._streaming_circuit_breaker:
            stats["circuit_breaker"] = self._streaming_circuit_breaker.get_stats()

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
        if self._streaming_circuit_breaker:
            self._streaming_circuit_breaker.reset()

    # Required abstract method implementations

    @property
    def source_type(self) -> SourceType:
        """The source type for this connector."""
        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable name for this connector."""
        return "RabbitMQ"

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[Evidence]:
        """
        Search is not supported for message queue connectors.

        RabbitMQ is a message queue - historical search requires
        specialized tooling or message archival systems.
        """
        logger.warning("[RabbitMQ] Search not supported for message queue connector")
        return []

    async def fetch(self, evidence_id: str) -> Evidence | None:
        """
        Fetch is not supported for message queue connectors.

        RabbitMQ messages are consumed once and not randomly accessible
        after acknowledgment.
        """
        logger.warning("[RabbitMQ] Fetch not supported for message queue connector")
        return None

    async def sync_items(
        self,
        state: SyncState,
        batch_size: int = 100,
    ) -> AsyncIterator[SyncItem]:
        """
        Yield items from RabbitMQ queue for incremental sync.

        Args:
            state: Sync state with cursor position
            batch_size: Number of messages to consume per batch

        Yields:
            SyncItem objects for Knowledge Mound ingestion
        """
        async for item in self.sync(batch_size=batch_size):
            yield item


__all__ = ["RabbitMQConnector", "RabbitMQConfig", "RabbitMQMessage"]
