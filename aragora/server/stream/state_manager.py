"""
Debate and loop state management for streaming.

NOTE: For general debate state management, prefer using aragora.server.state.StateManager:

    from aragora.server.state import get_state_manager
    state = get_state_manager()
    state.register_debate(...)

This module provides streaming-specific features:
- LoopInstance: Nomic loop tracking
- BoundedDebateDict: Memory-bounded debate tracking
- Periodic cleanup tasks for streaming state

The global debate tracking functions in this module delegate to the canonical
StateManager in aragora.server.state where applicable.
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from aragora.config import (
    MAX_ACTIVE_DEBATES,
    MAX_ACTIVE_LOOPS,
    MAX_DEBATE_STATES,
)

logger = logging.getLogger(__name__)


class BoundedDebateDict(OrderedDict):
    """OrderedDict with a maximum size, evicting oldest entries when full.

    Thread-safety must be provided externally via _active_debates_lock.
    """

    def __init__(self, maxsize: int = MAX_ACTIVE_DEBATES):
        super().__init__()
        self.maxsize = maxsize

    def __setitem__(self, key, value):
        # If key already exists, just update it
        if key in self:
            super().__setitem__(key, value)
            return
        # Evict oldest if at capacity
        while len(self) >= self.maxsize:
            oldest_key, oldest_val = self.popitem(last=False)
            logger.debug(f"Evicted oldest debate {oldest_key} to maintain maxsize={self.maxsize}")
        super().__setitem__(key, value)


@dataclass
class LoopInstance:
    """Represents an active nomic loop instance."""

    loop_id: str
    name: str
    started_at: float
    cycle: int = 0
    phase: str = "starting"
    path: str = ""


# =============================================================================
# Global debate tracking state
# =============================================================================

# Thread-safe debate tracking (bounded to prevent memory leaks)
_active_debates: BoundedDebateDict = BoundedDebateDict(maxsize=MAX_ACTIVE_DEBATES)
_active_debates_lock = threading.Lock()
_debate_executor: Optional[ThreadPoolExecutor] = None
_debate_executor_lock = threading.Lock()
_debate_cleanup_counter = 0
_debate_cleanup_counter_lock = threading.Lock()

# TTL for completed debates (24 hours)
_DEBATE_TTL_SECONDS = 86400


def get_active_debates() -> BoundedDebateDict:
    """Get the global active debates dictionary."""
    return _active_debates


def get_active_debates_lock() -> threading.Lock:
    """Get the lock for accessing active debates."""
    return _active_debates_lock


def get_debate_executor() -> Optional[ThreadPoolExecutor]:
    """Get the global debate executor."""
    global _debate_executor
    return _debate_executor


def set_debate_executor(executor: Optional[ThreadPoolExecutor]) -> None:
    """Set the global debate executor."""
    global _debate_executor
    _debate_executor = executor


def get_debate_executor_lock() -> threading.Lock:
    """Get the lock for accessing the debate executor."""
    return _debate_executor_lock


def cleanup_stale_debates() -> None:
    """Remove completed/errored debates older than TTL."""
    now = time.time()
    with _active_debates_lock:
        stale_ids = [
            debate_id
            for debate_id, debate in _active_debates.items()
            if debate.get("status") in ("completed", "error")
            and now - debate.get("completed_at", now) > _DEBATE_TTL_SECONDS
        ]
        for debate_id in stale_ids:
            _active_debates.pop(debate_id, None)
    if stale_ids:
        logger.debug(f"Cleaned up {len(stale_ids)} stale debate entries")


def increment_cleanup_counter() -> bool:
    """Increment cleanup counter and return True if cleanup should run."""
    global _debate_cleanup_counter
    with _debate_cleanup_counter_lock:
        _debate_cleanup_counter += 1
        if _debate_cleanup_counter >= 100:  # Every 100 operations
            _debate_cleanup_counter = 0
            return True
    return False


class DebateStateManager:
    """
    Manages debate and loop state with TTL-based cleanup.

    Provides thread-safe access to debate states, active loops,
    and cartographer instances with automatic cleanup.
    """

    def __init__(self):
        # Multi-loop tracking with TTL cleanup
        self.active_loops: dict[str, LoopInstance] = {}
        self._active_loops_lock = threading.Lock()
        self._active_loops_last_access: dict[str, float] = {}
        self._ACTIVE_LOOPS_TTL = 86400  # 24 hour TTL for stale loops
        self._MAX_ACTIVE_LOOPS = MAX_ACTIVE_LOOPS  # From config

        # Debate state caching with TTL cleanup
        self.debate_states: dict[str, dict] = {}
        self._debate_states_lock = threading.Lock()
        self._debate_states_last_access: dict[str, float] = {}
        self._DEBATE_STATES_TTL = 3600  # 1 hour TTL for ended debates
        self._MAX_DEBATE_STATES = MAX_DEBATE_STATES  # From config

        # Rate limiter tracking with cleanup (thread-safe counter)
        self._rate_limiter_cleanup_counter = 0
        self._cleanup_counter_lock = threading.Lock()
        self._CLEANUP_INTERVAL = 100  # Cleanup every N accesses

    def register_loop(self, loop_id: str, name: str, path: str = "") -> LoopInstance:
        """Register a new nomic loop instance."""
        with self._active_loops_lock:
            # Enforce max size with LRU eviction
            if len(self.active_loops) >= self._MAX_ACTIVE_LOOPS:
                oldest = min(
                    self._active_loops_last_access,
                    key=self._active_loops_last_access.get,
                    default=None,
                )
                if oldest:
                    self.active_loops.pop(oldest, None)
                    self._active_loops_last_access.pop(oldest, None)

            instance = LoopInstance(
                loop_id=loop_id,
                name=name,
                started_at=time.time(),
                path=path,
            )
            self.active_loops[loop_id] = instance
            self._active_loops_last_access[loop_id] = time.time()
            return instance

    def unregister_loop(self, loop_id: str) -> bool:
        """Unregister a nomic loop instance. Returns True if found."""
        with self._active_loops_lock:
            if loop_id in self.active_loops:
                del self.active_loops[loop_id]
                self._active_loops_last_access.pop(loop_id, None)
                return True
            return False

    def update_loop_state(
        self, loop_id: str, cycle: Optional[int] = None, phase: Optional[str] = None
    ) -> None:
        """Update the state of an active loop instance."""
        with self._active_loops_lock:
            if loop_id in self.active_loops:
                if cycle is not None:
                    self.active_loops[loop_id].cycle = cycle
                if phase is not None:
                    self.active_loops[loop_id].phase = phase
                self._active_loops_last_access[loop_id] = time.time()

    def get_loop_list(self) -> list[dict]:
        """Get list of active loops for client sync."""
        with self._active_loops_lock:
            return [
                {
                    "loop_id": loop.loop_id,
                    "name": loop.name,
                    "started_at": loop.started_at,
                    "cycle": loop.cycle,
                    "phase": loop.phase,
                    "path": loop.path,
                }
                for loop in self.active_loops.values()
            ]

    def get_debate_state(self, loop_id: str) -> Optional[dict]:
        """Get cached debate state for a loop."""
        with self._debate_states_lock:
            state = self.debate_states.get(loop_id)
            if state:
                self._debate_states_last_access[loop_id] = time.time()
            return state

    def set_debate_state(self, loop_id: str, state: dict) -> None:
        """Set cached debate state for a loop."""
        with self._debate_states_lock:
            # Enforce max size with LRU eviction (only evict ended debates)
            if len(self.debate_states) >= self._MAX_DEBATE_STATES:
                ended_states = [
                    (k, self._debate_states_last_access.get(k, 0))
                    for k, v in self.debate_states.items()
                    if v.get("ended")
                ]
                if ended_states:
                    oldest = min(ended_states, key=lambda x: x[1])[0]
                    self.debate_states.pop(oldest, None)
                    self._debate_states_last_access.pop(oldest, None)

            self.debate_states[loop_id] = state
            self._debate_states_last_access[loop_id] = time.time()

    def remove_debate_state(self, loop_id: str) -> None:
        """Remove cached debate state for a loop."""
        with self._debate_states_lock:
            self.debate_states.pop(loop_id, None)
            self._debate_states_last_access.pop(loop_id, None)

    def cleanup_stale_entries(self) -> int:
        """Remove stale entries from all tracking dicts. Returns count cleaned."""
        now = time.time()
        cleaned_count = 0

        # Cleanup active_loops older than TTL
        with self._active_loops_lock:
            stale = [
                k
                for k, v in self._active_loops_last_access.items()
                if now - v > self._ACTIVE_LOOPS_TTL
            ]
            for k in stale:
                self.active_loops.pop(k, None)
                self._active_loops_last_access.pop(k, None)
                cleaned_count += 1

        # Cleanup debate_states older than TTL (only ended debates)
        with self._debate_states_lock:
            stale = [
                k
                for k, state in self.debate_states.items()
                if state.get("ended")
                and now - self._debate_states_last_access.get(k, 0) > self._DEBATE_STATES_TTL
            ]
            for k in stale:
                self.debate_states.pop(k, None)
                self._debate_states_last_access.pop(k, None)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} stale entries")

        return cleaned_count

    def should_cleanup(self) -> bool:
        """Check if periodic cleanup should run (thread-safe counter)."""
        with self._cleanup_counter_lock:
            self._rate_limiter_cleanup_counter += 1
            if self._rate_limiter_cleanup_counter >= self._CLEANUP_INTERVAL:
                self._rate_limiter_cleanup_counter = 0
                return True
            return False


# =============================================================================
# Global DebateStateManager Singleton
# =============================================================================

_state_manager: Optional[DebateStateManager] = None
_state_manager_lock = threading.Lock()


def get_stream_state_manager() -> DebateStateManager:
    """Get the global DebateStateManager singleton for streaming.

    This returns the streaming-specific DebateStateManager which handles:
    - Loop instance tracking (LoopInstance)
    - Streaming debate state caching
    - Async cleanup tasks

    For general debate management, use aragora.server.state.get_state_manager() instead:

        from aragora.server.state import get_state_manager
        state = get_state_manager()  # Returns StateManager

    Thread-safe lazy initialization of the shared state manager instance.
    """
    global _state_manager
    if _state_manager is None:
        with _state_manager_lock:
            if _state_manager is None:
                _state_manager = DebateStateManager()
    return _state_manager


def get_state_manager() -> DebateStateManager:
    """DEPRECATED: Use get_stream_state_manager() for streaming state.

    For general debate management, use aragora.server.state.get_state_manager():

        from aragora.server.state import get_state_manager
        state = get_state_manager()

    This alias is kept for backward compatibility.
    """
    import warnings
    warnings.warn(
        "stream.state_manager.get_state_manager() is deprecated. "
        "Use get_stream_state_manager() for streaming or "
        "aragora.server.state.get_state_manager() for general debate management.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_stream_state_manager()


# =============================================================================
# Periodic Cleanup Background Task
# =============================================================================

_cleanup_task: Optional[asyncio.Task] = None
_default_cleanup_interval = 300  # 5 minutes


async def periodic_state_cleanup(
    manager: DebateStateManager,
    interval_seconds: int = 300,
) -> None:
    """Background task to periodically clean stale state entries.

    Runs indefinitely, cleaning up stale entries from active_loops and
    debate_states dictionaries every `interval_seconds`.

    This ensures cleanup happens even on low-traffic servers where the
    counter-based cleanup (every 100 operations) might never trigger.

    Args:
        manager: The DebateStateManager instance to clean
        interval_seconds: Seconds between cleanup runs (default: 300 = 5 min)
    """
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            cleaned = manager.cleanup_stale_entries()
            if cleaned > 0:
                logger.info(f"Periodic cleanup: removed {cleaned} stale state entries")
        except Exception as e:
            logger.warning(f"Periodic state cleanup error: {e}")


def start_cleanup_task(
    manager: DebateStateManager,
    interval_seconds: int = 300,
) -> asyncio.Task:
    """Start the periodic cleanup background task.

    Safe to call multiple times - will only start one task.

    Args:
        manager: The DebateStateManager instance to clean
        interval_seconds: Seconds between cleanup runs

    Returns:
        The asyncio.Task running the cleanup loop
    """
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(periodic_state_cleanup(manager, interval_seconds))
        logger.debug(f"Started periodic state cleanup task (interval={interval_seconds}s)")
    return _cleanup_task


def stop_cleanup_task() -> None:
    """Stop the periodic cleanup task gracefully.

    Safe to call even if no task is running.
    """
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        logger.debug("Stopped periodic state cleanup task")
    _cleanup_task = None


__all__ = [
    "BoundedDebateDict",
    "LoopInstance",
    "DebateStateManager",
    # Global state accessors
    "get_active_debates",
    "get_active_debates_lock",
    "get_debate_executor",
    "set_debate_executor",
    "get_debate_executor_lock",
    "get_stream_state_manager",  # Preferred - streaming-specific
    "get_state_manager",  # Deprecated - use get_stream_state_manager() instead
    "cleanup_stale_debates",
    "increment_cleanup_counter",
    # Periodic cleanup
    "periodic_state_cleanup",
    "start_cleanup_task",
    "stop_cleanup_task",
]
