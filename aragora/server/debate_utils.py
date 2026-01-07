"""
Ad-hoc debate management utilities.

Provides state tracking and helper functions for managing ad-hoc debates
started through the API.
"""

import logging
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

from aragora.server.stream import (
    SyncEventEmitter,
    StreamEvent,
    StreamEventType,
)
from aragora.server.error_utils import safe_error_message as _safe_error_message

logger = logging.getLogger(__name__)

# =============================================================================
# Global state for ad-hoc debates
# =============================================================================

# Track active ad-hoc debates
_active_debates: Dict[str, dict] = {}
_active_debates_lock = threading.Lock()  # Thread-safe access to _active_debates
_debate_cleanup_counter = 0  # Counter for periodic cleanup

# TTL for completed debates (24 hours)
_DEBATE_TTL_SECONDS = 86400


def get_active_debates() -> Dict[str, dict]:
    """Get the global active debates dictionary."""
    return _active_debates


def get_active_debates_lock() -> threading.Lock:
    """Get the lock for accessing active debates."""
    return _active_debates_lock


def update_debate_status(debate_id: str, status: str, **kwargs) -> None:
    """Atomic debate status update with consistent locking.

    Args:
        debate_id: The debate ID to update
        status: New status (e.g., "running", "completed", "error")
        **kwargs: Additional fields to update
    """
    with _active_debates_lock:
        if debate_id in _active_debates:
            _active_debates[debate_id]["status"] = status
            # Record completion time for TTL cleanup
            if status in ("completed", "error"):
                _active_debates[debate_id]["completed_at"] = time.time()
            for key, value in kwargs.items():
                _active_debates[debate_id][key] = value


def cleanup_stale_debates() -> None:
    """Remove completed/errored debates older than TTL."""
    now = time.time()
    with _active_debates_lock:
        stale_ids = [
            debate_id for debate_id, debate in _active_debates.items()
            if debate.get("status") in ("completed", "error")
            and now - debate.get("completed_at", now) > _DEBATE_TTL_SECONDS
        ]
        for debate_id in stale_ids:
            _active_debates.pop(debate_id, None)
    if stale_ids:
        logger.debug(f"Cleaned up {len(stale_ids)} stale debate entries")


def increment_cleanup_counter() -> bool:
    """Increment cleanup counter and return True if cleanup should run.

    Cleanup runs every 100 debates to avoid frequent expensive operations.
    """
    global _debate_cleanup_counter
    _debate_cleanup_counter += 1
    if _debate_cleanup_counter >= 100:
        _debate_cleanup_counter = 0
        return True
    return False


def wrap_agent_for_streaming(
    agent: Any,
    emitter: SyncEventEmitter,
    debate_id: str
) -> Any:
    """Wrap an agent to emit token streaming events.

    If the agent has a generate_stream() method, we override its generate()
    to call generate_stream() and emit TOKEN_* events.

    Args:
        agent: Agent instance (duck-typed, must have generate method)
        emitter: Event emitter for streaming events
        debate_id: ID of the current debate

    Returns:
        The agent with wrapped generate method (or unchanged if no streaming support)
    """
    # Check if agent supports streaming
    if not hasattr(agent, 'generate_stream'):
        return agent

    # Store original generate method
    original_generate = agent.generate

    async def streaming_generate(prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Streaming wrapper that emits TOKEN_* events."""
        # Emit start event
        emitter.emit(StreamEvent(
            type=StreamEventType.TOKEN_START,
            data={
                "debate_id": debate_id,
                "agent": agent.name,
                "timestamp": datetime.now().isoformat(),
            },
            agent=agent.name,
        ))

        full_response = ""
        try:
            # Stream tokens from the agent
            async for token in agent.generate_stream(prompt, context):
                full_response += token
                # Emit token delta event
                emitter.emit(StreamEvent(
                    type=StreamEventType.TOKEN_DELTA,
                    data={
                        "debate_id": debate_id,
                        "agent": agent.name,
                        "token": token,
                    },
                    agent=agent.name,
                ))

            # Emit end event
            emitter.emit(StreamEvent(
                type=StreamEventType.TOKEN_END,
                data={
                    "debate_id": debate_id,
                    "agent": agent.name,
                    "full_response": full_response,
                },
                agent=agent.name,
            ))

            return full_response

        except Exception as e:
            # Emit error as end event
            emitter.emit(StreamEvent(
                type=StreamEventType.TOKEN_END,
                data={
                    "debate_id": debate_id,
                    "agent": agent.name,
                    "error": _safe_error_message(e, f"token streaming for {agent.name}"),
                    "full_response": full_response,
                },
                agent=agent.name,
            ))
            # Fall back to non-streaming
            if full_response:
                return full_response
            return await original_generate(prompt, context)

    # Replace the generate method
    agent.generate = streaming_generate
    return agent


# Backward compatibility aliases (prefixed with underscore)
_update_debate_status = update_debate_status
_cleanup_stale_debates = cleanup_stale_debates
_wrap_agent_for_streaming = wrap_agent_for_streaming


__all__ = [
    # State accessors
    "get_active_debates",
    "get_active_debates_lock",
    # Functions
    "update_debate_status",
    "cleanup_stale_debates",
    "increment_cleanup_counter",
    "wrap_agent_for_streaming",
    # Constants
    "_DEBATE_TTL_SECONDS",
    # Backward compatibility (underscore-prefixed)
    "_active_debates",
    "_active_debates_lock",
    "_update_debate_status",
    "_cleanup_stale_debates",
    "_wrap_agent_for_streaming",
]
