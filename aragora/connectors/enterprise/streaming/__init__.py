"""
Enterprise Streaming Connectors.

Real-time event stream ingestion from message brokers:
- Kafka: Apache Kafka topics with consumer group management
- RabbitMQ: AMQP message queues with acknowledgment handling

These connectors enable omnivorous real-time data ingestion into
Aragora's Knowledge Mound and decision pipelines.

Resilience features:
- Connection retry with exponential backoff
- Circuit breaker for broker failure protection
- Dead letter queue (DLQ) for failed messages
- Graceful shutdown on SIGTERM
- Health monitoring
"""

from aragora.connectors.enterprise.streaming.kafka import (
    KafkaConnector,
    KafkaConfig,
)
from aragora.connectors.enterprise.streaming.rabbitmq import (
    RabbitMQConnector,
    RabbitMQConfig,
)
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

__all__ = [
    # Connectors
    "KafkaConnector",
    "KafkaConfig",
    "RabbitMQConnector",
    "RabbitMQConfig",
    # Resilience
    "StreamingResilienceConfig",
    "ExponentialBackoff",
    "CircuitState",
    "CircuitBreakerOpenError",
    "StreamingCircuitBreaker",
    "DLQMessage",
    "DLQHandler",
    "GracefulShutdown",
    "HealthStatus",
    "HealthMonitor",
    "with_retry",
]
