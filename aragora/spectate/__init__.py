"""Spectator Mode - Real-time debate observation."""

from .events import SpectatorEvents
from .stream import SpectatorStream
from .ws_bridge import (
    SpectateEvent,
    SpectateWebSocketBridge,
    bind_spectate_context,
    get_spectate_bridge,
)

__all__ = [
    "SpectatorEvents",
    "SpectatorStream",
    "SpectateEvent",
    "SpectateWebSocketBridge",
    "bind_spectate_context",
    "get_spectate_bridge",
]
