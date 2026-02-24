"""
Streaming reliability module.

Provides connection hardening with automatic reconnection for WebSocket
and enterprise streaming (Kafka, RabbitMQ) connections.

Usage:
    from aragora.streaming.reliability import (
        ReconnectPolicy,
        ReliableConnection,
        ReliableWebSocket,
        ReliableKafkaConsumer,
        ConnectionState,
        ConnectionQualityMetrics,
    )
"""

from .reliability import (
    ConnectionQualityMetrics,
    ConnectionState,
    ReconnectPolicy,
    ReliableConnection,
    ReliableKafkaConsumer,
    ReliableWebSocket,
)

__all__ = [
    "ConnectionQualityMetrics",
    "ConnectionState",
    "ReconnectPolicy",
    "ReliableConnection",
    "ReliableKafkaConsumer",
    "ReliableWebSocket",
]
