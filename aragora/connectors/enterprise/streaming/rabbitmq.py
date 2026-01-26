"""
RabbitMQ Enterprise Connector.

Message queue ingestion from RabbitMQ with:
- Exchange and queue binding management
- Message acknowledgment for reliable delivery
- Dead letter queue handling
- JSON and binary message deserialization

Requires: aio-pika

Usage:
    config = RabbitMQConfig(
        url="amqp://guest:guest@localhost/",
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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Dict, Optional

from aragora.connectors.base import Evidence
from aragora.connectors.enterprise.base import (
    EnterpriseConnector,
    SyncItem,
    SyncState,
)
from aragora.reasoning.provenance import SourceType

logger = logging.getLogger(__name__)


@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ connector."""

    # Connection
    url: str = "amqp://guest:guest@localhost/"
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
    dead_letter_exchange: Optional[str] = None
    dead_letter_routing_key: Optional[str] = None
    message_ttl: Optional[int] = None  # milliseconds

    # SSL/TLS
    ssl: bool = False
    ssl_cafile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    # Processing
    batch_size: int = 100
    auto_ack: bool = False  # Manual ack for reliability
    requeue_on_error: bool = True
    message_handler: Optional[Callable] = None


@dataclass
class RabbitMQMessage:
    """A RabbitMQ message with metadata."""

    queue: str
    routing_key: str
    delivery_tag: int
    body: Any  # Deserialized payload
    headers: Dict[str, str]
    timestamp: datetime
    message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expiration: Optional[str] = None
    priority: Optional[int] = None

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

    Uses aio-pika for async operation.
    """

    def __init__(self, config: RabbitMQConfig, **kwargs):
        """
        Initialize RabbitMQ connector.

        Args:
            config: RabbitMQConfig with connection and processing settings
        """
        super().__init__(connector_id="rabbitmq", **kwargs)
        self.config = config
        self._connection = None
        self._channel = None
        self._queue = None
        self._running = False
        self._consumed_count = 0
        self._error_count = 0
        self._acked_count = 0
        self._nacked_count = 0

    async def connect(self) -> bool:
        """
        Connect to RabbitMQ server.

        Returns:
            True if connection successful
        """
        try:
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

            # Create channel with QoS
            self._channel = await self._connection.channel()  # type: ignore[attr-defined]
            await self._channel.set_qos(prefetch_count=self.config.prefetch_count)  # type: ignore[attr-defined]

            # Declare exchange if specified
            if self.config.exchange:
                exchange = await self._channel.declare_exchange(  # type: ignore[attr-defined]
                    self.config.exchange,
                    type=self.config.exchange_type,
                    durable=self.config.durable,
                )
            else:
                exchange = self._channel.default_exchange  # type: ignore[attr-defined]

            # Build queue arguments
            queue_args = {}
            if self.config.dead_letter_exchange:
                queue_args["x-dead-letter-exchange"] = self.config.dead_letter_exchange
            if self.config.dead_letter_routing_key:
                queue_args["x-dead-letter-routing-key"] = self.config.dead_letter_routing_key
            if self.config.message_ttl:
                queue_args["x-message-ttl"] = self.config.message_ttl  # type: ignore[assignment]

            # Declare queue
            self._queue = await self._channel.declare_queue(  # type: ignore[attr-defined]
                self.config.queue,
                durable=self.config.durable,
                auto_delete=self.config.auto_delete,
                exclusive=self.config.exclusive,
                arguments=queue_args or None,
            )

            # Bind queue to exchange if specified
            if self.config.exchange:
                await self._queue.bind(  # type: ignore[attr-defined]
                    exchange,
                    routing_key=self.config.routing_key or self.config.queue,
                )

            logger.info(
                f"[RabbitMQ] Connected to {self.config.url}, "
                f"queue={self.config.queue}, exchange={self.config.exchange or 'default'}"
            )
            return True

        except ImportError:
            logger.error("[RabbitMQ] aio-pika not installed. Install with: pip install aio-pika")
            return False
        except Exception as e:
            logger.error(f"[RabbitMQ] Connection failed: {e}")
            return False

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
        """Start consuming messages."""
        if not self._connection:
            connected = await self.connect()
            if not connected:
                raise RuntimeError("Failed to connect to RabbitMQ")
        self._running = True

    async def stop(self) -> None:
        """Stop consuming messages."""
        await self.disconnect()

    async def consume(self, max_messages: Optional[int] = None) -> AsyncIterator[RabbitMQMessage]:
        """
        Consume messages from RabbitMQ queue.

        Args:
            max_messages: Optional limit on messages to consume

        Yields:
            RabbitMQMessage objects
        """
        if not self._queue:
            await self.start()

        messages_consumed = 0

        try:
            async with self._queue.iterator() as queue_iter:  # type: ignore[attr-defined]
                async for message in queue_iter:
                    try:
                        # Deserialize message
                        rabbitmq_msg = self._deserialize_message(message)
                        yield rabbitmq_msg

                        self._consumed_count += 1
                        messages_consumed += 1

                        # Auto-ack if configured
                        if self.config.auto_ack:
                            await message.ack()
                            self._acked_count += 1

                        if max_messages and messages_consumed >= max_messages:
                            break

                    except Exception as e:
                        self._error_count += 1
                        logger.warning(f"[RabbitMQ] Error processing message: {e}")
                        if self.config.requeue_on_error:
                            await message.nack(requeue=True)
                        else:
                            await message.reject(requeue=False)
                        self._nacked_count += 1

        except asyncio.CancelledError:
            logger.info("[RabbitMQ] Consumer cancelled")
            raise

    def _deserialize_message(self, message) -> RabbitMQMessage:
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
            # Auto-ack after successful conversion
            if not self.config.auto_ack:
                await msg.ack()
                self._acked_count += 1

    async def publish(
        self,
        body: Any,
        routing_key: str = None,
        headers: Dict[str, str] = None,
        message_id: str = None,
        correlation_id: str = None,
        reply_to: str = None,
        expiration: str = None,
        priority: int = None,
    ) -> bool:
        """
        Publish a message to RabbitMQ.

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

            # Get exchange
            if self.config.exchange:
                exchange = await self._channel.get_exchange(self.config.exchange)  # type: ignore[attr-defined]
            else:
                exchange = self._channel.default_exchange  # type: ignore[attr-defined]

            # Publish
            await exchange.publish(
                message,
                routing_key=routing_key or self.config.routing_key or self.config.queue,
            )

            logger.debug(f"[RabbitMQ] Published message to {routing_key or self.config.queue}")
            return True

        except Exception as e:
            logger.error(f"[RabbitMQ] Failed to publish message: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "connector_id": self.connector_id,
            "url": self.config.url.split("@")[-1],  # Hide credentials
            "queue": self.config.queue,
            "exchange": self.config.exchange or "default",
            "running": self._running,
            "consumed_count": self._consumed_count,
            "acked_count": self._acked_count,
            "nacked_count": self._nacked_count,
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
        return "RabbitMQ"

    async def search(
        self,
        query: str,
        limit: int = 10,
        **kwargs,
    ) -> list[Evidence]:
        """
        Search is not supported for message queue connectors.

        RabbitMQ is a message queue - historical search requires
        specialized tooling or message archival systems.
        """
        logger.warning("[RabbitMQ] Search not supported for message queue connector")
        return []

    async def fetch(self, evidence_id: str) -> Optional[Evidence]:
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
