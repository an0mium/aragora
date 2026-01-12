"""
Stream package for real-time debate streaming via WebSocket.

This package provides the infrastructure for broadcasting debate events
to connected clients in real-time.

Main components:
- events: Event types and data classes (StreamEventType, StreamEvent)
- emitter: Event emitter and audience participation (SyncEventEmitter, AudienceInbox)
- state_manager: Debate and loop state management (DebateStateManager)
- arena_hooks: Arena integration hooks (create_arena_hooks)
- broadcaster: Client management and broadcasting utilities (WebSocketBroadcaster)
- server_base: Common server functionality (ServerBase, ServerConfig)
- servers: WebSocket and HTTP server classes (DebateStreamServer, AiohttpUnifiedServer)
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    # Events
    "StreamEventType": ("aragora.server.stream.events", "StreamEventType"),
    "StreamEvent": ("aragora.server.stream.events", "StreamEvent"),
    "AudienceMessage": ("aragora.server.stream.events", "AudienceMessage"),
    # Emitter
    "TokenBucket": ("aragora.server.stream.emitter", "TokenBucket"),
    "AudienceInbox": ("aragora.server.stream.emitter", "AudienceInbox"),
    "SyncEventEmitter": ("aragora.server.stream.emitter", "SyncEventEmitter"),
    "normalize_intensity": ("aragora.server.stream.emitter", "normalize_intensity"),
    # State management
    "BoundedDebateDict": ("aragora.server.stream.state_manager", "BoundedDebateDict"),
    "LoopInstance": ("aragora.server.stream.state_manager", "LoopInstance"),
    "DebateStateManager": ("aragora.server.stream.state_manager", "DebateStateManager"),
    "get_active_debates": ("aragora.server.stream.state_manager", "get_active_debates"),
    "get_active_debates_lock": ("aragora.server.stream.state_manager", "get_active_debates_lock"),
    "get_debate_executor": ("aragora.server.stream.state_manager", "get_debate_executor"),
    "set_debate_executor": ("aragora.server.stream.state_manager", "set_debate_executor"),
    "get_debate_executor_lock": ("aragora.server.stream.state_manager", "get_debate_executor_lock"),
    "cleanup_stale_debates": ("aragora.server.stream.state_manager", "cleanup_stale_debates"),
    "increment_cleanup_counter": ("aragora.server.stream.state_manager", "increment_cleanup_counter"),
    # Arena hooks
    "create_arena_hooks": ("aragora.server.stream.arena_hooks", "create_arena_hooks"),
    "wrap_agent_for_streaming": ("aragora.server.stream.arena_hooks", "wrap_agent_for_streaming"),
    # Gauntlet streaming
    "GauntletStreamEmitter": ("aragora.server.stream.gauntlet_emitter", "GauntletStreamEmitter"),
    "GauntletPhase": ("aragora.server.stream.gauntlet_emitter", "GauntletPhase"),
    "create_gauntlet_emitter": ("aragora.server.stream.gauntlet_emitter", "create_gauntlet_emitter"),
    # Broadcaster
    "BroadcasterConfig": ("aragora.server.stream.broadcaster", "BroadcasterConfig"),
    "ClientManager": ("aragora.server.stream.broadcaster", "ClientManager"),
    "DebateStateCache": ("aragora.server.stream.broadcaster", "DebateStateCache"),
    "LoopRegistry": ("aragora.server.stream.broadcaster", "LoopRegistry"),
    "WebSocketBroadcaster": ("aragora.server.stream.broadcaster", "WebSocketBroadcaster"),
    # Server base
    "ServerBase": ("aragora.server.stream.server_base", "ServerBase"),
    "ServerConfig": ("aragora.server.stream.server_base", "ServerConfig"),
    # Servers
    "DebateStreamServer": ("aragora.server.stream.servers", "DebateStreamServer"),
    "AiohttpUnifiedServer": ("aragora.server.stream.servers", "AiohttpUnifiedServer"),
    "DEBATE_AVAILABLE": ("aragora.server.stream.servers", "DEBATE_AVAILABLE"),
    # Backward compatibility
    "_cleanup_stale_debates_stream": ("aragora.server.stream.servers", "_cleanup_stale_debates_stream"),
    "_wrap_agent_for_streaming": ("aragora.server.stream.servers", "_wrap_agent_for_streaming"),
    "_active_debates": ("aragora.server.stream.state_manager", "_active_debates"),
    "_active_debates_lock": ("aragora.server.stream.state_manager", "_active_debates_lock"),
    "_debate_executor_lock": ("aragora.server.stream.state_manager", "_debate_executor_lock"),
    "_DEBATE_TTL_SECONDS": ("aragora.server.stream.servers", "_DEBATE_TTL_SECONDS"),
    "TRUSTED_PROXIES": ("aragora.server.stream.servers", "TRUSTED_PROXIES"),
    "_safe_error_message": ("aragora.server.error_utils", "safe_error_message"),
    "_debate_executor": ("aragora.server.stream.state_manager", "_debate_executor"),
    "_get_active_debates": ("aragora.server.stream.state_manager", "get_active_debates"),
}

_DYNAMIC_EXPORTS = {"_debate_executor"}

__all__ = [
    # Events
    "StreamEventType",
    "StreamEvent",
    "AudienceMessage",
    # Emitter
    "TokenBucket",
    "AudienceInbox",
    "SyncEventEmitter",
    "normalize_intensity",
    # State management
    "BoundedDebateDict",
    "LoopInstance",
    "DebateStateManager",
    "get_active_debates",
    "get_active_debates_lock",
    "get_debate_executor",
    "set_debate_executor",
    "get_debate_executor_lock",
    "cleanup_stale_debates",
    "increment_cleanup_counter",
    # Arena hooks
    "create_arena_hooks",
    "wrap_agent_for_streaming",
    # Broadcaster
    "BroadcasterConfig",
    "ClientManager",
    "DebateStateCache",
    "LoopRegistry",
    "WebSocketBroadcaster",
    # Server base
    "ServerBase",
    "ServerConfig",
    # Servers
    "DebateStreamServer",
    "AiohttpUnifiedServer",
    "DEBATE_AVAILABLE",
    # Backward compatibility
    "_cleanup_stale_debates_stream",
    "_wrap_agent_for_streaming",
    "_active_debates",
    "_active_debates_lock",
    "_debate_executor_lock",
    "_DEBATE_TTL_SECONDS",
    "TRUSTED_PROXIES",
    "_safe_error_message",
    "_debate_executor",
]


def __getattr__(name: str) -> Any:
    """Lazily import stream components to avoid side effects on package import."""
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        if name not in _DYNAMIC_EXPORTS:
            globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
