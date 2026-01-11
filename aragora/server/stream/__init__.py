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

# Re-export main classes for backward compatibility
from .events import (
    StreamEventType,
    StreamEvent,
    AudienceMessage,
)
from .emitter import (
    TokenBucket,
    AudienceInbox,
    SyncEventEmitter,
    normalize_intensity,
)
from .state_manager import (
    BoundedDebateDict,
    LoopInstance,
    DebateStateManager,
    get_active_debates,
    get_active_debates_lock,
    get_debate_executor,
    set_debate_executor,
    get_debate_executor_lock,
    cleanup_stale_debates,
    increment_cleanup_counter,
)
from .arena_hooks import (
    create_arena_hooks,
    wrap_agent_for_streaming,
)
from .gauntlet_emitter import (
    GauntletStreamEmitter,
    GauntletPhase,
    create_gauntlet_emitter,
)
from .broadcaster import (
    BroadcasterConfig,
    ClientManager,
    DebateStateCache,
    LoopRegistry,
    WebSocketBroadcaster,
)
from .server_base import (
    ServerBase,
    ServerConfig,
)
from .servers import (
    DebateStreamServer,
    AiohttpUnifiedServer,
    DEBATE_AVAILABLE,
    _cleanup_stale_debates_stream,
    _wrap_agent_for_streaming,
    _active_debates,
    _active_debates_lock,
    _debate_executor_lock,
    _DEBATE_TTL_SECONDS,
    TRUSTED_PROXIES,
)

# Import error_utils for backward compatibility
from aragora.server.error_utils import safe_error_message as _safe_error_message

# Expose _debate_executor for backward compatibility (tests access it directly)
# NOTE: Use get_debate_executor() and set_debate_executor() in new code
from .state_manager import get_active_debates as _get_active_debates

# The _debate_executor is accessed via get/set functions now, but tests may expect direct access
# We expose it via a module-level reference that syncs with the state_manager
import aragora.server.stream.state_manager as _state_manager
_debate_executor = _state_manager._debate_executor

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
