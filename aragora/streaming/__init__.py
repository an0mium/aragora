"""
Streaming reliability module.

Provides connection hardening with automatic reconnection for WebSocket
and enterprise streaming (Kafka, RabbitMQ) connections, plus production-
grade circuit breakers, replay buffers, health monitoring, and
reconnection management.

Usage:
    from aragora.streaming.reliability import (
        ReconnectPolicy,
        ReliableConnection,
        ReliableWebSocket,
        ReliableKafkaConsumer,
        ConnectionState,
        ConnectionQualityMetrics,
    )
    from aragora.streaming.circuit_breaker import StreamCircuitBreaker
    from aragora.streaming.replay_buffer import EventReplayBuffer
    from aragora.streaming.health_monitor import get_stream_health_monitor
    from aragora.streaming.reconnection import ReconnectionManager
"""

from .reliability import (
    ConnectionQualityMetrics,
    ConnectionState,
    ReconnectPolicy,
    ReliableConnection,
    ReliableKafkaConsumer,
    ReliableWebSocket,
)

from .circuit_breaker import (
    StreamCircuitBreaker,
    StreamCircuitBreakerConfig,
    StreamCircuitState,
)

from .replay_buffer import (
    BufferedEvent,
    EventReplayBuffer,
)

from .health_monitor import (
    StreamHealthMonitor,
    StreamHealthSnapshot,
    get_stream_health_monitor,
)

from .reconnection import (
    ConnectionQualityScore,
    ReconnectionConfig,
    ReconnectionContext,
    ReconnectionManager,
)

__all__ = [
    # reliability.py
    "ConnectionQualityMetrics",
    "ConnectionState",
    "ReconnectPolicy",
    "ReliableConnection",
    "ReliableKafkaConsumer",
    "ReliableWebSocket",
    # circuit_breaker.py
    "StreamCircuitBreaker",
    "StreamCircuitBreakerConfig",
    "StreamCircuitState",
    # replay_buffer.py
    "BufferedEvent",
    "EventReplayBuffer",
    # health_monitor.py
    "StreamHealthMonitor",
    "StreamHealthSnapshot",
    "get_stream_health_monitor",
    # reconnection.py
    "ConnectionQualityScore",
    "ReconnectionConfig",
    "ReconnectionContext",
    "ReconnectionManager",
]
