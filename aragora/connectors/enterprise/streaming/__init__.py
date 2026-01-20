"""
Enterprise Streaming Connectors.

Real-time event stream ingestion from message brokers:
- Kafka: Apache Kafka topics with consumer group management
- RabbitMQ: AMQP message queues with acknowledgment handling

These connectors enable omnivorous real-time data ingestion into
Aragora's Knowledge Mound and decision pipelines.
"""

from aragora.connectors.enterprise.streaming.kafka import (
    KafkaConnector,
    KafkaConfig,
)
from aragora.connectors.enterprise.streaming.rabbitmq import (
    RabbitMQConnector,
    RabbitMQConfig,
)

__all__ = [
    "KafkaConnector",
    "KafkaConfig",
    "RabbitMQConnector",
    "RabbitMQConfig",
]
