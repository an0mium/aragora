"""
Aragora SDK for Python.

Official SDK for the Aragora multi-agent debate platform.

Example:
    >>> from aragora import AragoraClient
    >>>
    >>> async with AragoraClient(api_key="your-key") as client:
    ...     debate = await client.create_debate(
    ...         question="Should we use microservices?",
    ...         agents=["claude", "gpt-4"],
    ...     )
    ...     print(debate["debate_id"])

For synchronous usage:
    >>> from aragora import AragoraClientSync
    >>>
    >>> client = AragoraClientSync(api_key="your-key")
    >>> debate = client.create_debate(
    ...     question="Should we use microservices?",
    ...     agents=["claude", "gpt-4"],
    ... )
    >>> client.close()
"""

from aragora.client import (
    ApiConfig,
    AragoraClient,
    AragoraClientSync,
)
from aragora.streaming import (
    AragoraWebSocket,
    WebSocketEvent,
    WebSocketOptions,
    WebSocketState,
    stream_debate,
    stream_debate_by_id,
)

__version__ = "2.4.0"
__all__ = [
    # Client
    "AragoraClient",
    "AragoraClientSync",
    "ApiConfig",
    # Streaming
    "AragoraWebSocket",
    "WebSocketEvent",
    "WebSocketOptions",
    "WebSocketState",
    "stream_debate",
    "stream_debate_by_id",
    # Version
    "__version__",
]
