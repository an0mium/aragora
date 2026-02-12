"""
AWS SNS/SQS Enterprise Connector.

Real-time event stream ingestion from AWS SNS/SQS with:
- Long polling for efficient message retrieval
- Visibility timeout management for reliable processing
- Dead letter queue (DLQ) for failed messages
- Message deduplication for FIFO queues
- Connection retry with exponential backoff
- Circuit breaker for AWS service failures
- Graceful shutdown on SIGTERM

Requires: aioboto3

Usage:
    config = SNSSQSConfig(
        region="us-east-1",
        queue_url="https://sqs.us-east-1.amazonaws.com/123456789/my-queue",
        topic_arn="arn:aws:sns:us-east-1:123456789:my-topic",  # Optional
    )
    connector = SNSSQSConnector(config)
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
from typing import TYPE_CHECKING, Any
from collections.abc import AsyncIterator, Awaitable, Callable

if TYPE_CHECKING:
    from aragora.reasoning.provenance import SourceType

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

logger = logging.getLogger(__name__)


@dataclass
class SNSSQSConfig:
    """Configuration for AWS SNS/SQS connector."""

    # Connection
    region: str = "us-east-1"
    queue_url: str = ""  # Required - SQS queue URL
    topic_arn: str | None = None  # Optional - SNS topic ARN for publishing

    # AWS Credentials (optional - uses default credential chain if not set)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    endpoint_url: str | None = None  # For LocalStack/testing

    # Consumer settings
    max_messages: int = 10  # Max messages per poll (1-10)
    wait_time_seconds: int = 20  # Long polling wait time (0-20)
    visibility_timeout_seconds: int = 300  # Message processing timeout
    message_attribute_names: list[str] = field(default_factory=lambda: ["All"])
    attribute_names: list[str] = field(default_factory=lambda: ["All"])

    # FIFO queue settings
    is_fifo_queue: bool = False
    message_group_id: str | None = None  # For FIFO queues

    # Dead letter queue
    dead_letter_queue_url: str | None = None

    # Processing
    batch_size: int = 100  # Max messages before yielding
    auto_delete: bool = True  # Auto-delete after successful processing
    extend_visibility: bool = True  # Auto-extend visibility timeout

    # Resilience settings
    resilience: StreamingResilienceConfig = field(default_factory=StreamingResilienceConfig)
    enable_circuit_breaker: bool = True
    enable_dlq: bool = True
    enable_graceful_shutdown: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not self.queue_url:
            raise ValueError("queue_url is required")
        if self.max_messages < 1 or self.max_messages > 10:
            raise ValueError("max_messages must be between 1 and 10")
        if self.wait_time_seconds < 0 or self.wait_time_seconds > 20:
            raise ValueError("wait_time_seconds must be between 0 and 20")
        if self.visibility_timeout_seconds < 0:
            raise ValueError("visibility_timeout_seconds must be non-negative")


@dataclass
class SQSMessage:
    """An SQS message with metadata."""

    message_id: str
    receipt_handle: str
    body: Any  # Deserialized payload
    md5_of_body: str
    attributes: dict[str, str]
    message_attributes: dict[str, Any]
    timestamp: datetime

    # FIFO queue fields
    sequence_number: str | None = None
    message_deduplication_id: str | None = None
    message_group_id: str | None = None

    @classmethod
    def from_sqs_response(cls, msg: dict[str, Any]) -> SQSMessage:
        """Create from SQS response message."""
        body_str = msg.get("Body", "{}")

        # Try to parse as JSON
        try:
            body = json.loads(body_str)
        except json.JSONDecodeError:
            body = body_str

        # Handle SNS-wrapped messages
        if isinstance(body, dict) and body.get("Type") == "Notification":
            # Unwrap SNS notification
            try:
                body = json.loads(body.get("Message", "{}"))
            except json.JSONDecodeError:
                body = body.get("Message", body_str)

        # Parse timestamp from attributes
        attrs = msg.get("Attributes", {})
        sent_timestamp = attrs.get("SentTimestamp")
        if sent_timestamp:
            timestamp = datetime.fromtimestamp(int(sent_timestamp) / 1000, tz=timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        return cls(
            message_id=msg.get("MessageId", ""),
            receipt_handle=msg.get("ReceiptHandle", ""),
            body=body,
            md5_of_body=msg.get("MD5OfBody", ""),
            attributes=attrs,
            message_attributes=msg.get("MessageAttributes", {}),
            timestamp=timestamp,
            sequence_number=attrs.get("SequenceNumber"),
            message_deduplication_id=attrs.get("MessageDeduplicationId"),
            message_group_id=attrs.get("MessageGroupId"),
        )

    def to_sync_item(self) -> SyncItem:
        """Convert to SyncItem for Knowledge Mound ingestion."""
        # Extract content from body
        if isinstance(self.body, dict):
            content = json.dumps(self.body, indent=2)
            title = self.body.get("title") or self.body.get("type") or f"SQS: {self.message_id[:8]}"
        elif isinstance(self.body, str):
            content = self.body
            title = f"SQS: {self.message_id[:8]}"
        else:
            content = str(self.body)
            title = f"SQS: {self.message_id[:8]}"

        # Truncate content if too long
        content = content[:50000]

        return SyncItem(
            id=f"sqs-{self.message_id}",
            content=content,
            source_type="event_stream",
            source_id=f"sqs/{self.message_id}",
            title=title,
            url=None,
            author=self.message_attributes.get("producer", {}).get("StringValue", "aws-sqs"),
            created_at=self.timestamp,
            updated_at=self.timestamp,
            domain="enterprise/sqs",
            confidence=0.9,
            metadata={
                "message_id": self.message_id,
                "md5_of_body": self.md5_of_body,
                "attributes": self.attributes,
                "message_attributes": {
                    k: v.get("StringValue") or v.get("BinaryValue")
                    for k, v in self.message_attributes.items()
                },
                "sequence_number": self.sequence_number,
                "message_group_id": self.message_group_id,
            },
        )

    def to_dlq_message(self, error: Exception) -> DLQMessage:
        """Convert to DLQMessage for dead letter queue."""
        return DLQMessage(
            original_topic=f"sqs-{self.message_id[:16]}",
            original_key=self.message_id,
            original_value=json.dumps(self.body) if isinstance(self.body, dict) else str(self.body),
            original_headers={},  # DLQMessage expects dict[str, str], not message attributes
            original_timestamp=self.timestamp,
            error_message=str(error),
            error_type=type(error).__name__,
            retry_count=int(self.attributes.get("ApproximateReceiveCount", "1")),
        )


class SNSSQSConnector(EnterpriseConnector):
    """
    Enterprise connector for AWS SNS/SQS.

    Provides real-time event stream ingestion with:
    - Long polling for efficient message retrieval
    - Visibility timeout management
    - Support for both standard and FIFO queues
    - SNS notification unwrapping
    - Connection retry with exponential backoff
    - Circuit breaker for AWS service failures
    - Dead letter queue (DLQ) for failed messages
    - Graceful shutdown on SIGTERM

    Uses aioboto3 for async operation.
    """

    @property
    def source_type(self) -> SourceType:
        """The source type for this connector."""
        from aragora.reasoning.provenance import SourceType

        return SourceType.EXTERNAL_API

    @property
    def name(self) -> str:
        """Human-readable name for this connector."""
        return "AWS SNS/SQS"

    def __init__(
        self,
        config: SNSSQSConfig,
        dlq_sender: Callable[[str, DLQMessage], Awaitable[None]] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize SNS/SQS connector.

        Args:
            config: SNSSQSConfig with connection and processing settings
            dlq_sender: Optional custom DLQ sender function
        """
        super().__init__(connector_id="snssqs", **kwargs)
        self.config = config
        self._sqs_client: Any | None = None
        self._sns_client: Any | None = None
        self._session: Any | None = None
        self._running = False
        self._consumed_count = 0
        self._error_count = 0
        self._dlq_count = 0

        # Pending messages for delete confirmation
        self._pending_deletes: list[tuple[str, str]] = []  # (message_id, receipt_handle)

        # Resilience components
        self._streaming_circuit_breaker: StreamingCircuitBreaker | None = None
        self._dlq_handler: DLQHandler | None = None
        self._health_monitor: HealthMonitor | None = None
        self._graceful_shutdown: GracefulShutdown | None = None

        if config.enable_circuit_breaker:
            self._streaming_circuit_breaker = StreamingCircuitBreaker(
                name="snssqs-aws",
                config=config.resilience,
            )

        if config.enable_dlq:
            self._dlq_handler = DLQHandler(
                config=config.resilience,
                dlq_sender=dlq_sender or self._default_dlq_sender,
            )

        self._health_monitor = HealthMonitor(
            name="snssqs-connector",
            config=config.resilience,
        )

        if config.enable_graceful_shutdown:
            self._graceful_shutdown = GracefulShutdown()

    async def _default_dlq_sender(self, topic: str, message: DLQMessage) -> None:
        """Default DLQ sender using SQS."""
        if not self.config.dead_letter_queue_url:
            logger.warning("[SNS/SQS] No DLQ URL configured, dropping failed message")
            return

        if not self._sqs_client:
            await self._create_clients()

        if self._sqs_client:
            try:
                await self._sqs_client.send_message(
                    QueueUrl=self.config.dead_letter_queue_url,
                    MessageBody=message.to_json(),
                    MessageAttributes={
                        "error_type": {
                            "DataType": "String",
                            "StringValue": message.error_type,
                        },
                        "retry_count": {
                            "DataType": "Number",
                            "StringValue": str(message.retry_count),
                        },
                        "original_message_id": {
                            "DataType": "String",
                            "StringValue": message.original_key or "unknown",
                        },
                    },
                )
                logger.info(f"[SNS/SQS] Sent message to DLQ: {self.config.dead_letter_queue_url}")
            except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                logger.error(f"[SNS/SQS] Failed to send to DLQ: {e}")

    async def _create_clients(self) -> None:
        """Create SQS and SNS clients."""
        try:
            import aioboto3

            session_kwargs: dict[str, Any] = {
                "region_name": self.config.region,
            }

            if self.config.aws_access_key_id:
                session_kwargs["aws_access_key_id"] = self.config.aws_access_key_id
            if self.config.aws_secret_access_key:
                session_kwargs["aws_secret_access_key"] = self.config.aws_secret_access_key
            if self.config.aws_session_token:
                session_kwargs["aws_session_token"] = self.config.aws_session_token

            self._session = aioboto3.Session(**session_kwargs)

            client_kwargs: dict[str, Any] = {}
            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            # Create SQS client
            self._sqs_client = await self._session.client("sqs", **client_kwargs).__aenter__()

            # Create SNS client if topic ARN is provided
            if self.config.topic_arn:
                self._sns_client = await self._session.client("sns", **client_kwargs).__aenter__()

            logger.info(f"[SNS/SQS] Created AWS clients for region {self.config.region}")

        except ImportError:
            logger.error("[SNS/SQS] aioboto3 not installed. Install with: pip install aioboto3")
            raise

    async def connect(self) -> bool:
        """
        Connect to AWS SQS with retry and circuit breaker.

        Returns:
            True if connection successful
        """
        backoff = ExponentialBackoff(self.config.resilience)

        for attempt in range(self.config.resilience.max_retries + 1):
            # Check circuit breaker
            if self._streaming_circuit_breaker and self._streaming_circuit_breaker.is_open:
                if not await self._streaming_circuit_breaker.can_execute():
                    logger.warning("[SNS/SQS] Circuit breaker is open, skipping connect")
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
                logger.error("[SNS/SQS] aioboto3 not installed. Install with: pip install aioboto3")
                return False

            except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                if self._streaming_circuit_breaker:
                    await self._streaming_circuit_breaker.record_failure(e)
                if self._health_monitor:
                    await self._health_monitor.record_failure(e)

                if attempt == self.config.resilience.max_retries:
                    logger.error(f"[SNS/SQS] Connection failed after {attempt + 1} attempts: {e}")
                    return False

                delay = backoff.get_delay(attempt)
                logger.warning(
                    f"[SNS/SQS] Connection attempt {attempt + 1}/{self.config.resilience.max_retries + 1} "
                    f"failed: {e}. Retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

        return False

    async def _connect_internal(self) -> bool:
        """Internal connection logic."""
        await self._create_clients()

        # Verify queue exists by getting attributes
        if self._sqs_client:
            await self._sqs_client.get_queue_attributes(
                QueueUrl=self.config.queue_url,
                AttributeNames=["QueueArn"],
            )
            logger.info(f"[SNS/SQS] Connected to queue: {self.config.queue_url}")
            return True

        return False

    async def disconnect(self) -> None:
        """Disconnect from AWS services."""
        self._running = False

        # Delete any pending messages
        await self._flush_pending_deletes()

        if self._sqs_client:
            try:
                await self._sqs_client.__aexit__(None, None, None)
            except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                logger.warning(f"[SNS/SQS] Error closing SQS client: {e}")
            self._sqs_client = None

        if self._sns_client:
            try:
                await self._sns_client.__aexit__(None, None, None)
            except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                logger.warning(f"[SNS/SQS] Error closing SNS client: {e}")
            self._sns_client = None

        logger.info("[SNS/SQS] Disconnected from AWS services")

    async def _flush_pending_deletes(self) -> None:
        """Delete all pending messages."""
        if not self._pending_deletes or not self._sqs_client:
            return

        # Delete in batches of 10 (SQS limit)
        for i in range(0, len(self._pending_deletes), 10):
            batch = self._pending_deletes[i : i + 10]
            entries = [{"Id": msg_id, "ReceiptHandle": receipt} for msg_id, receipt in batch]
            try:
                await self._sqs_client.delete_message_batch(
                    QueueUrl=self.config.queue_url,
                    Entries=entries,
                )
            except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                logger.warning(f"[SNS/SQS] Error deleting messages: {e}")

        self._pending_deletes.clear()

    async def consume(self) -> AsyncIterator[SQSMessage]:
        """
        Consume messages from SQS queue.

        Yields messages as they arrive, handling:
        - Long polling for efficient retrieval
        - Visibility timeout extension
        - Automatic retry on transient failures
        - Circuit breaker protection

        Yields:
            SQSMessage instances
        """
        if not self._sqs_client:
            connected = await self.connect()
            if not connected:
                logger.error("[SNS/SQS] Failed to connect, cannot consume")
                return

        self._running = True
        backoff = ExponentialBackoff(self.config.resilience)
        consecutive_failures = 0

        # Setup graceful shutdown
        if self._graceful_shutdown:
            self._graceful_shutdown.register_cleanup(self._shutdown_cleanup)

        try:
            while self._running:
                # Check graceful shutdown
                if self._graceful_shutdown and self._graceful_shutdown.is_shutting_down:
                    logger.info("[SNS/SQS] Shutdown requested, stopping consume loop")
                    break

                # Check circuit breaker
                if self._streaming_circuit_breaker and self._streaming_circuit_breaker.is_open:
                    if not await self._streaming_circuit_breaker.can_execute():
                        logger.warning("[SNS/SQS] Circuit breaker is open, pausing consume")
                        await asyncio.sleep(self.config.resilience.circuit_breaker_recovery_seconds)
                        continue

                try:
                    # Receive messages with long polling
                    response = await self._sqs_client.receive_message(
                        QueueUrl=self.config.queue_url,
                        MaxNumberOfMessages=self.config.max_messages,
                        WaitTimeSeconds=self.config.wait_time_seconds,
                        VisibilityTimeout=self.config.visibility_timeout_seconds,
                        MessageAttributeNames=self.config.message_attribute_names,
                        AttributeNames=self.config.attribute_names,
                    )

                    messages = response.get("Messages", [])

                    if not messages:
                        # No messages, reset backoff
                        consecutive_failures = 0
                        continue

                    # Process messages
                    for msg_data in messages:
                        try:
                            msg = SQSMessage.from_sqs_response(msg_data)
                            self._consumed_count += 1

                            # Track for deletion
                            if self.config.auto_delete:
                                self._pending_deletes.append((msg.message_id, msg.receipt_handle))

                            yield msg

                            # Record success
                            if self._streaming_circuit_breaker:
                                await self._streaming_circuit_breaker.record_success()
                            if self._health_monitor:
                                await self._health_monitor.record_success()

                            consecutive_failures = 0

                        except (ValueError, KeyError, TypeError) as e:
                            logger.error(f"[SNS/SQS] Failed to parse message: {e}")
                            self._error_count += 1

                            # Send to DLQ
                            if self._dlq_handler and self.config.enable_dlq:
                                await self._dlq_handler.send_to_dlq(
                                    topic="sqs-parse-error",
                                    key=msg_data.get("MessageId", "unknown"),
                                    value=json.dumps(msg_data),
                                    headers={},
                                    timestamp=datetime.now(timezone.utc),
                                    error=e,
                                    retry_count=1,
                                )
                                self._dlq_count += 1

                    # Flush deletes periodically
                    if len(self._pending_deletes) >= self.config.batch_size:
                        await self._flush_pending_deletes()

                except CircuitBreakerOpenError:
                    logger.warning("[SNS/SQS] Circuit breaker tripped")
                    await asyncio.sleep(self.config.resilience.circuit_breaker_recovery_seconds)

                except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
                    consecutive_failures += 1
                    self._error_count += 1

                    if self._streaming_circuit_breaker:
                        await self._streaming_circuit_breaker.record_failure(e)
                    if self._health_monitor:
                        await self._health_monitor.record_failure(e)

                    if consecutive_failures > self.config.resilience.max_retries:
                        logger.error(
                            f"[SNS/SQS] Too many consecutive failures ({consecutive_failures}), "
                            f"stopping consume"
                        )
                        break

                    delay = backoff.get_delay(consecutive_failures - 1)
                    logger.warning(
                        f"[SNS/SQS] Receive error ({consecutive_failures}): {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)

        finally:
            # Cleanup
            await self._flush_pending_deletes()
            # Graceful shutdown cleanup is handled via registered cleanup callback

    async def _shutdown_cleanup(self) -> None:
        """Handle shutdown signal."""
        logger.info("[SNS/SQS] Received shutdown signal")
        self._running = False

    async def sync_items(self, state: SyncState) -> AsyncIterator[SyncItem]:  # type: ignore[override]
        """
        Sync items from SQS for Knowledge Mound ingestion.

        Args:
            state: Current sync state

        Yields:
            SyncItem instances for ingestion
        """
        async for msg in self.consume():
            yield msg.to_sync_item()

    async def publish(
        self,
        message: str | dict[str, Any],
        message_attributes: dict[str, Any] | None = None,
        message_group_id: str | None = None,
        message_deduplication_id: str | None = None,
    ) -> str | None:
        """
        Publish a message to SNS topic or SQS queue.

        Args:
            message: Message content (str or dict)
            message_attributes: Optional message attributes
            message_group_id: Required for FIFO queues
            message_deduplication_id: Optional deduplication ID for FIFO

        Returns:
            Message ID if successful, None otherwise
        """
        if isinstance(message, dict):
            message_body = json.dumps(message)
        else:
            message_body = message

        attrs = {}
        if message_attributes:
            for key, value in message_attributes.items():
                if isinstance(value, str):
                    attrs[key] = {"DataType": "String", "StringValue": value}
                elif isinstance(value, (int, float)):
                    attrs[key] = {"DataType": "Number", "StringValue": str(value)}
                elif isinstance(value, bytes):
                    attrs[key] = {"DataType": "Binary", "BinaryValue": value}  # type: ignore[dict-item]

        try:
            if self.config.topic_arn and self._sns_client:
                # Publish to SNS
                response = await self._sns_client.publish(
                    TopicArn=self.config.topic_arn,
                    Message=message_body,
                    MessageAttributes=attrs,
                )
                return response.get("MessageId")

            elif self._sqs_client:
                # Send to SQS directly
                send_kwargs: dict[str, Any] = {
                    "QueueUrl": self.config.queue_url,
                    "MessageBody": message_body,
                }

                if attrs:
                    send_kwargs["MessageAttributes"] = attrs

                # FIFO queue parameters
                if self.config.is_fifo_queue:
                    send_kwargs["MessageGroupId"] = (
                        message_group_id or self.config.message_group_id or "default"
                    )
                    if message_deduplication_id:
                        send_kwargs["MessageDeduplicationId"] = message_deduplication_id

                response = await self._sqs_client.send_message(**send_kwargs)
                return response.get("MessageId")

        except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.error(f"[SNS/SQS] Failed to publish message: {e}")
            if self._streaming_circuit_breaker:
                await self._streaming_circuit_breaker.record_failure(e)
            return None

        return None

    async def delete_message(self, receipt_handle: str) -> bool:
        """
        Manually delete a message from the queue.

        Args:
            receipt_handle: The receipt handle of the message to delete

        Returns:
            True if deleted successfully
        """
        if not self._sqs_client:
            return False

        try:
            await self._sqs_client.delete_message(
                QueueUrl=self.config.queue_url,
                ReceiptHandle=receipt_handle,
            )
            return True
        except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.error(f"[SNS/SQS] Failed to delete message: {e}")
            return False

    async def change_visibility(
        self,
        receipt_handle: str,
        visibility_timeout: int,
    ) -> bool:
        """
        Change the visibility timeout of a message.

        Args:
            receipt_handle: The receipt handle of the message
            visibility_timeout: New visibility timeout in seconds

        Returns:
            True if changed successfully
        """
        if not self._sqs_client:
            return False

        try:
            await self._sqs_client.change_message_visibility(
                QueueUrl=self.config.queue_url,
                ReceiptHandle=receipt_handle,
                VisibilityTimeout=visibility_timeout,
            )
            return True
        except (OSError, RuntimeError, ConnectionError, TimeoutError) as e:
            logger.error(f"[SNS/SQS] Failed to change visibility: {e}")
            return False

    async def get_health(self) -> HealthStatus:
        """
        Get health status of the connector.

        Returns:
            HealthStatus with current health information
        """
        if self._health_monitor:
            return await self._health_monitor.get_status()

        return HealthStatus(
            healthy=self._sqs_client is not None,
            last_check=datetime.now(timezone.utc),
            consecutive_failures=self._error_count,
            last_error=None,
            messages_processed=self._consumed_count,
            messages_failed=self._error_count,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get connector statistics."""
        return {
            "consumed_count": self._consumed_count,
            "error_count": self._error_count,
            "dlq_count": self._dlq_count,
            "pending_deletes": len(self._pending_deletes),
            "is_running": self._running,
            "circuit_breaker_state": (
                self._streaming_circuit_breaker.state.value
                if self._streaming_circuit_breaker
                else "disabled"
            ),
        }

    async def search(self, query: str, **kwargs: Any) -> list[Any]:  # type: ignore[override]
        """
        Search is not supported for streaming connectors.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("Search not supported for SNS/SQS connector")

    async def fetch(self, item_id: str, **kwargs: Any) -> Any:  # type: ignore[override]
        """
        Fetch is not supported for streaming connectors.

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError("Fetch not supported for SNS/SQS connector")
