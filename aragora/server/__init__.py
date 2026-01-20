"""
aragora.server - Live debate streaming and permalink sharing.

Components:
- stream: WebSocket server for real-time debate broadcasts
- storage: SQLite-backed permalink persistence
- api: REST API for debate retrieval
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "DebateStreamServer": ("aragora.server.stream", "DebateStreamServer"),
    "SyncEventEmitter": ("aragora.server.stream", "SyncEventEmitter"),
    "StreamEvent": ("aragora.server.stream", "StreamEvent"),
    "StreamEventType": ("aragora.server.stream", "StreamEventType"),
    "create_arena_hooks": ("aragora.server.stream", "create_arena_hooks"),
    "DebateStorage": ("aragora.server.storage", "DebateStorage"),
    "DebateMetadata": ("aragora.server.storage", "DebateMetadata"),
    "run_api_server": ("aragora.server.api", "run_api_server"),
}
# Note: Prometheus metrics (prometheus_nomic, prometheus_control_plane, prometheus_rlm)
# are internal infrastructure - import directly from submodules to avoid circular imports.

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    """Lazily import server components to avoid side effects on package import."""
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
