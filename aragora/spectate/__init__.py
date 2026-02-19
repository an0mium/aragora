"""Spectator Mode - Real-time debate observation."""

from .events import SpectatorEvents
from .stream import SpectatorStream
from .ws_bridge import SpectateEvent, SpectateWebSocketBridge, get_spectate_bridge

__all__ = [
    "SpectatorEvents",
    "SpectatorStream",
    "SpectateEvent",
    "SpectateWebSocketBridge",
    "get_spectate_bridge",
]
