"""
Type definitions and protocols for Aragora.

Provides structural typing via protocols to enable
duck typing while maintaining type safety.
"""

from aragora.types.protocols import (
    EventEmitterProtocol,
    EventHandlerProtocol,
    SyncEventHandlerProtocol,
)

__all__ = [
    "EventEmitterProtocol",
    "EventHandlerProtocol",
    "SyncEventHandlerProtocol",
]
