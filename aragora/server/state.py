"""
Centralized server state management.

Consolidates global state that was previously scattered across
stream.py and unified_server.py to prevent inconsistencies.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class DebateState:
    """State for an active debate."""
    debate_id: str
    task: str
    agents: list[str]
    start_time: float
    status: str = "running"
    current_round: int = 0
    total_rounds: int = 3
    messages: list = field(default_factory=list)
    subscribers: Set[Any] = field(default_factory=set)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "debate_id": self.debate_id,
            "task": self.task,
            "agents": self.agents,
            "start_time": self.start_time,
            "status": self.status,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "message_count": len(self.messages),
            "subscriber_count": len(self.subscribers),
            "elapsed_seconds": time.time() - self.start_time,
        }


class StateManager:
    """
    Thread-safe centralized state manager for the server.

    Consolidates:
    - Active debates tracking
    - Thread pool executor management
    - Cleanup counters
    - Server start time

    Usage:
        state = get_state_manager()
        state.register_debate("debate-123", {...})
        debates = state.get_active_debates()
    """

    def __init__(self):
        """Initialize state manager."""
        # Active debates
        self._active_debates: Dict[str, DebateState] = {}
        self._debates_lock = threading.Lock()

        # Thread pool executor for debates
        self._executor: Optional[ThreadPoolExecutor] = None
        self._executor_lock = threading.Lock()
        self._executor_max_workers = 4

        # Cleanup tracking
        self._cleanup_counter = 0
        self._cleanup_counter_lock = threading.Lock()
        self._cleanup_interval = 10  # Run cleanup every N debates

        # Server metadata
        self._server_start_time = time.time()
        self._shutdown_callbacks: list[Callable] = []

    @property
    def server_start_time(self) -> float:
        """Get server start timestamp."""
        return self._server_start_time

    @property
    def uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self._server_start_time

    # ==================== Debate Management ====================

    def register_debate(
        self,
        debate_id: str,
        task: str,
        agents: list[str],
        total_rounds: int = 3,
        metadata: Optional[dict] = None,
    ) -> DebateState:
        """
        Register a new active debate.

        Args:
            debate_id: Unique debate identifier
            task: The debate task/topic
            agents: List of participating agent names
            total_rounds: Total number of debate rounds
            metadata: Optional additional metadata

        Returns:
            The created DebateState
        """
        state = DebateState(
            debate_id=debate_id,
            task=task,
            agents=agents,
            start_time=time.time(),
            total_rounds=total_rounds,
            metadata=metadata or {},
        )

        with self._debates_lock:
            self._active_debates[debate_id] = state
            logger.debug(f"Registered debate {debate_id}, total active: {len(self._active_debates)}")

        return state

    def unregister_debate(self, debate_id: str) -> Optional[DebateState]:
        """
        Unregister a debate when it completes.

        Args:
            debate_id: Debate identifier to remove

        Returns:
            The removed DebateState, or None if not found
        """
        with self._debates_lock:
            state = self._active_debates.pop(debate_id, None)
            if state:
                logger.debug(f"Unregistered debate {debate_id}, remaining: {len(self._active_debates)}")

        # Trigger periodic cleanup
        self._maybe_cleanup()

        return state

    def get_debate(self, debate_id: str) -> Optional[DebateState]:
        """Get a debate's state by ID."""
        with self._debates_lock:
            return self._active_debates.get(debate_id)

    def get_active_debates(self) -> Dict[str, DebateState]:
        """Get a copy of all active debates."""
        with self._debates_lock:
            return dict(self._active_debates)

    def get_active_debate_count(self) -> int:
        """Get count of active debates."""
        with self._debates_lock:
            return len(self._active_debates)

    def update_debate_status(
        self,
        debate_id: str,
        status: Optional[str] = None,
        current_round: Optional[int] = None,
    ) -> bool:
        """
        Update a debate's status.

        Args:
            debate_id: Debate to update
            status: New status (e.g., "running", "completed", "failed")
            current_round: Current round number

        Returns:
            True if update succeeded, False if debate not found
        """
        with self._debates_lock:
            state = self._active_debates.get(debate_id)
            if state is None:
                return False

            if status is not None:
                state.status = status
            if current_round is not None:
                state.current_round = current_round

            return True

    def add_debate_message(self, debate_id: str, message: Any) -> bool:
        """
        Add a message to a debate's history.

        Args:
            debate_id: Debate to update
            message: Message to add

        Returns:
            True if added, False if debate not found
        """
        with self._debates_lock:
            state = self._active_debates.get(debate_id)
            if state is None:
                return False

            state.messages.append(message)
            return True

    def add_subscriber(self, debate_id: str, subscriber: Any) -> bool:
        """Add a subscriber (WebSocket) to a debate."""
        with self._debates_lock:
            state = self._active_debates.get(debate_id)
            if state is None:
                return False

            state.subscribers.add(subscriber)
            return True

    def remove_subscriber(self, debate_id: str, subscriber: Any) -> bool:
        """Remove a subscriber from a debate."""
        with self._debates_lock:
            state = self._active_debates.get(debate_id)
            if state is None:
                return False

            state.subscribers.discard(subscriber)
            return True

    # ==================== Executor Management ====================

    def get_executor(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """
        Get or create the shared thread pool executor.

        Args:
            max_workers: Optional max workers (only used on first call)

        Returns:
            The thread pool executor
        """
        with self._executor_lock:
            if self._executor is None:
                workers = max_workers or self._executor_max_workers
                self._executor = ThreadPoolExecutor(
                    max_workers=workers,
                    thread_name_prefix="debate-"
                )
                logger.info(f"Created debate executor with {workers} workers")

            return self._executor

    def shutdown_executor(self, wait: bool = True) -> None:
        """
        Shutdown the thread pool executor.

        Args:
            wait: Whether to wait for pending tasks
        """
        with self._executor_lock:
            if self._executor is not None:
                logger.info("Shutting down debate executor")
                self._executor.shutdown(wait=wait)
                self._executor = None

    # ==================== Cleanup ====================

    def _maybe_cleanup(self) -> None:
        """Run periodic cleanup if counter threshold reached."""
        with self._cleanup_counter_lock:
            self._cleanup_counter += 1
            if self._cleanup_counter < self._cleanup_interval:
                return
            self._cleanup_counter = 0

        self._cleanup_stale_debates()

    def _cleanup_stale_debates(self, max_age_seconds: float = 3600) -> int:
        """
        Remove debates that have been running too long.

        Args:
            max_age_seconds: Maximum age before considering stale

        Returns:
            Number of debates cleaned up
        """
        now = time.time()
        stale_ids = []

        with self._debates_lock:
            for debate_id, state in self._active_debates.items():
                if now - state.start_time > max_age_seconds:
                    stale_ids.append(debate_id)

        for debate_id in stale_ids:
            logger.warning(f"Cleaning up stale debate {debate_id}")
            self.unregister_debate(debate_id)

        return len(stale_ids)

    # ==================== Shutdown ====================

    def register_shutdown_callback(self, callback: Callable) -> None:
        """Register a callback to run on shutdown."""
        self._shutdown_callbacks.append(callback)

    def shutdown(self) -> None:
        """
        Shutdown the state manager and cleanup resources.

        Runs registered shutdown callbacks and cleans up executor.
        """
        logger.info("StateManager shutdown initiated")

        # Run callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback error: {e}")

        # Shutdown executor
        self.shutdown_executor(wait=True)

        # Clear debates
        with self._debates_lock:
            self._active_debates.clear()

        logger.info("StateManager shutdown complete")

    # ==================== Stats ====================

    def get_stats(self) -> dict:
        """Get current state statistics."""
        with self._debates_lock:
            active_count = len(self._active_debates)
            statuses = {}
            for state in self._active_debates.values():
                statuses[state.status] = statuses.get(state.status, 0) + 1

        return {
            "active_debates": active_count,
            "debate_statuses": statuses,
            "uptime_seconds": self.uptime_seconds,
            "executor_active": self._executor is not None,
            "cleanup_counter": self._cleanup_counter,
        }


# Singleton instance
_state_manager: Optional[StateManager] = None
_state_manager_lock = threading.Lock()


def get_state_manager() -> StateManager:
    """
    Get the singleton StateManager instance.

    Thread-safe initialization.
    """
    global _state_manager

    if _state_manager is None:
        with _state_manager_lock:
            if _state_manager is None:
                _state_manager = StateManager()

    return _state_manager


def reset_state_manager() -> None:
    """
    Reset the state manager (for testing).

    Warning: Only use in tests!
    """
    global _state_manager

    with _state_manager_lock:
        if _state_manager is not None:
            _state_manager.shutdown()
        _state_manager = None
