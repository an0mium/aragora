"""
Ad-hoc debate management utilities.

Provides state tracking and helper functions for managing ad-hoc debates
started through the API.

Note: This module maintains backward compatibility with existing code that
imports global state variables. New code should use StateManager directly
via get_state_manager().
"""

import logging
import threading
import time
import warnings
from typing import Any, Dict, Optional

from aragora.server.state import get_state_manager
from aragora.server.stream import SyncEventEmitter

logger = logging.getLogger(__name__)

# =============================================================================
# Backward Compatibility Layer
# =============================================================================
# These globals are maintained for backward compatibility with existing code.
# They delegate to the centralized StateManager.
#
# New code should use:
#   from aragora.server.state import get_state_manager
#   state = get_state_manager()
#   state.register_debate(...)

# TTL for completed debates (24 hours)
_DEBATE_TTL_SECONDS = 86400


class _ActiveDebatesProxy(Dict[str, dict]):
    """Proxy dict that delegates to StateManager for backward compatibility.

    This allows existing code that accesses _active_debates directly to
    continue working while actually using the centralized StateManager.
    """

    def __getitem__(self, key: str) -> dict:
        state = get_state_manager().get_debate(key)
        if state is None:
            raise KeyError(key)
        return state.to_dict()

    def __setitem__(self, key: str, value: dict) -> None:
        # For direct assignment, register as new debate
        manager = get_state_manager()
        if manager.get_debate(key) is None:
            manager.register_debate(
                debate_id=key,
                task=value.get("task", ""),
                agents=value.get("agents", []),
                total_rounds=value.get("total_rounds", 3),
                metadata=value,
            )
        else:
            # Update existing
            manager.update_debate_status(
                key,
                status=value.get("status"),
                current_round=value.get("current_round"),
            )

    def __delitem__(self, key: str) -> None:
        get_state_manager().unregister_debate(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return get_state_manager().get_debate(key) is not None

    def __iter__(self):
        return iter(get_state_manager().get_active_debates().keys())

    def __len__(self) -> int:
        return get_state_manager().get_active_debate_count()

    def get(self, key: str, default: Any = None) -> Any:
        state = get_state_manager().get_debate(key)
        if state is None:
            return default
        return state.to_dict()

    def items(self):
        debates = get_state_manager().get_active_debates()
        return [(k, v.to_dict()) for k, v in debates.items()]

    def keys(self):
        return get_state_manager().get_active_debates().keys()

    def values(self):
        debates = get_state_manager().get_active_debates()
        return [v.to_dict() for v in debates.values()]

    def pop(self, key: str, *args) -> Optional[dict]:
        state = get_state_manager().unregister_debate(key)
        if state is not None:
            return state.to_dict()
        if args:
            return args[0]
        raise KeyError(key)


# Backward compatibility globals - delegate to StateManager
_active_debates: Dict[str, dict] = _ActiveDebatesProxy()
_active_debates_lock = threading.Lock()  # Kept for interface compatibility
_debate_cleanup_counter = 0  # Kept for interface compatibility


def get_active_debates() -> Dict[str, dict]:
    """Get the active debates dictionary.

    Returns a dict-like object that delegates to StateManager.
    For new code, prefer using get_state_manager().get_active_debates().
    """
    return _active_debates


def get_active_debates_lock() -> threading.Lock:
    """Get the lock for accessing active debates.

    Note: StateManager handles locking internally, so this lock is
    primarily for backward compatibility with code that expects it.
    """
    return _active_debates_lock


def update_debate_status(debate_id: str, status: str, **kwargs) -> None:
    """Atomic debate status update with consistent locking.

    Args:
        debate_id: The debate ID to update
        status: New status (e.g., "running", "completed", "error")
        **kwargs: Additional fields to update
    """
    manager = get_state_manager()
    state = manager.get_debate(debate_id)
    if state is not None:
        manager.update_debate_status(
            debate_id,
            status=status,
            current_round=kwargs.get("current_round"),
        )
        # Store additional kwargs in metadata
        if kwargs:
            state.metadata.update(kwargs)
        # Record completion time for TTL cleanup
        if status in ("completed", "error"):
            state.metadata["completed_at"] = time.time()


def cleanup_stale_debates() -> None:
    """Remove completed/errored debates older than TTL.

    Delegates to StateManager's internal cleanup mechanism.
    """
    manager = get_state_manager()
    now = time.time()

    # Get all debates and filter stale ones
    debates = manager.get_active_debates()
    stale_ids = []

    for debate_id, state in debates.items():
        if state.status in ("completed", "error"):
            completed_at = state.metadata.get("completed_at", state.start_time)
            if now - completed_at > _DEBATE_TTL_SECONDS:
                stale_ids.append(debate_id)

    for debate_id in stale_ids:
        manager.unregister_debate(debate_id)

    if stale_ids:
        logger.debug(f"Cleaned up {len(stale_ids)} stale debate entries")


def increment_cleanup_counter() -> bool:
    """Increment cleanup counter and return True if cleanup should run.

    Cleanup runs every 100 debates to avoid frequent expensive operations.

    Note: StateManager handles its own cleanup, but this is kept for
    backward compatibility with code that calls this explicitly.
    """
    # StateManager handles cleanup internally, so we can delegate
    # Just return False to indicate no external cleanup needed
    return False


def wrap_agent_for_streaming(agent: Any, emitter: SyncEventEmitter, debate_id: str) -> Any:
    """DEPRECATED: Use aragora.server.stream.wrap_agent_for_streaming instead.

    This function was missing task_id on TOKEN events, causing text interleaving
    when multiple agents stream concurrently. The correct implementation in
    aragora.server.stream.arena_hooks includes proper task_id handling.

    Args:
        agent: Agent instance (duck-typed, must have generate method)
        emitter: Event emitter for streaming events
        debate_id: ID of the current debate

    Returns:
        The agent with wrapped generate method (or unchanged if no streaming support)
    """
    warnings.warn(
        "debate_utils.wrap_agent_for_streaming is deprecated and was causing text "
        "interleaving bugs. Use aragora.server.stream.wrap_agent_for_streaming instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Import here to avoid circular imports
    from aragora.server.stream.arena_hooks import (
        wrap_agent_for_streaming as _correct_wrap,
    )

    return _correct_wrap(agent, emitter, debate_id)


# Backward compatibility aliases (prefixed with underscore)
_update_debate_status = update_debate_status
_cleanup_stale_debates = cleanup_stale_debates
_wrap_agent_for_streaming = wrap_agent_for_streaming


# Stuck debate timeout (10 minutes for production, prevents indefinitely running debates)
STUCK_DEBATE_TIMEOUT_SECONDS = 600


async def watchdog_stuck_debates(check_interval: float = 60.0) -> None:
    """Background coroutine to cleanup stuck debates.

    Runs periodically to check for debates that have been running too long
    without progress. Marks them as timed out and emits events.

    Args:
        check_interval: Seconds between checks (default 60s)
    """
    import asyncio

    from aragora.server.stream import StreamEvent, StreamEventType

    logger.info("[watchdog] Stuck debate watchdog started")

    while True:
        try:
            await asyncio.sleep(check_interval)

            now = time.time()
            stuck_debates = []
            manager = get_state_manager()

            # Get all active debates and find stuck ones
            debates = manager.get_active_debates()
            for debate_id, state in debates.items():
                if state.status in ("running", "starting"):
                    elapsed = now - state.start_time
                    if elapsed > STUCK_DEBATE_TIMEOUT_SECONDS:
                        stuck_debates.append((debate_id, elapsed))

            # Process stuck debates
            for debate_id, elapsed in stuck_debates:
                logger.warning(
                    f"[watchdog] Cancelling stuck debate {debate_id} "
                    f"(running for {elapsed:.0f}s, timeout: {STUCK_DEBATE_TIMEOUT_SECONDS}s)"
                )

                # Update status
                update_debate_status(
                    debate_id,
                    "timeout",
                    error=f"Debate timed out after {elapsed:.0f}s",
                )
                manager.update_debate_status(debate_id, status="timeout")

                # Try to cancel the task if tracked
                state = manager.get_debate(debate_id)
                if state:
                    task = state.metadata.get("_task")
                    if (
                        task
                        and hasattr(task, "cancel")
                        and not getattr(task, "done", lambda: True)()
                    ):
                        try:
                            task.cancel()
                        except Exception as e:
                            logger.debug(f"[watchdog] Task cancel failed for {debate_id}: {e}")

                # Emit timeout event
                try:
                    from aragora.server.stream import get_event_emitter

                    emitter = get_event_emitter()
                    if emitter:
                        emitter.emit(
                            StreamEvent(
                                type=StreamEventType.DEBATE_END,
                                data={
                                    "debate_id": debate_id,
                                    "status": "timeout",
                                    "reason": f"Debate timed out after {elapsed:.0f}s",
                                    "duration": elapsed,
                                },
                                loop_id=debate_id,
                            )
                        )
                except Exception as e:
                    logger.debug(f"[watchdog] Event emission failed for {debate_id}: {e}")

            if stuck_debates:
                logger.info(f"[watchdog] Cleaned up {len(stuck_debates)} stuck debate(s)")

        except asyncio.CancelledError:
            logger.info("[watchdog] Stuck debate watchdog cancelled")
            break
        except Exception as e:
            logger.error(f"[watchdog] Error in stuck debate watchdog: {e}")
            # Continue running despite errors


__all__ = [
    # State accessors
    "get_active_debates",
    "get_active_debates_lock",
    # Functions
    "update_debate_status",
    "cleanup_stale_debates",
    "increment_cleanup_counter",
    "wrap_agent_for_streaming",
    "watchdog_stuck_debates",
    # Constants
    "_DEBATE_TTL_SECONDS",
    "STUCK_DEBATE_TIMEOUT_SECONDS",
    # Backward compatibility (underscore-prefixed)
    "_active_debates",
    "_active_debates_lock",
    "_update_debate_status",
    "_cleanup_stale_debates",
    "_wrap_agent_for_streaming",
]
