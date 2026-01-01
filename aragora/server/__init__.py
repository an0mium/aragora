"""
aragora.server - Live debate streaming and permalink sharing.

Components:
- stream: WebSocket server for real-time debate broadcasts
- storage: SQLite-backed permalink persistence
- api: REST API for debate retrieval
"""

from aragora.server.stream import (
    DebateStreamServer,
    SyncEventEmitter,
    StreamEvent,
    StreamEventType,
    create_arena_hooks,
)
from aragora.server.storage import DebateStorage, DebateMetadata
from aragora.server.api import run_api_server

__all__ = [
    "DebateStreamServer",
    "SyncEventEmitter",
    "StreamEvent",
    "StreamEventType",
    "create_arena_hooks",
    "DebateStorage",
    "DebateMetadata",
    "run_api_server",
]
