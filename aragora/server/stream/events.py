"""
Stream event types and data classes.

Defines the types of events emitted during debates and nomic loop execution,
along with the dataclasses for representing events and audience messages.

NOTE: The event types are now defined in aragora.events.types to avoid
layering violations. This module re-exports them for backward compatibility.
"""

# Re-export from shared events layer for backward compatibility
from aragora.events.types import (
    AudienceMessage,
    StreamEvent,
    StreamEventType,
)

__all__ = [
    "StreamEventType",
    "StreamEvent",
    "AudienceMessage",
]
