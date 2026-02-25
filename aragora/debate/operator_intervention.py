"""
Operator Intervention Controls for Debate Management.

Provides pause, resume, restart, and context injection capabilities
for running debates through an in-memory registry. Uses asyncio.Event
to signal pause/resume to the debate loop between phases.

Usage:
    from aragora.debate.operator_intervention import (
        DebateInterventionManager,
        get_operator_manager,
    )

    # Register a debate when it starts
    manager = get_operator_manager()
    manager.register(debate_id, total_rounds=5)

    # Operator pauses the debate
    manager.pause(debate_id, reason="Reviewing intermediate results")

    # The debate loop checks between phases:
    #   await manager.wait_if_paused(debate_id)

    # Operator resumes
    manager.resume(debate_id)

    # Restart from a specific round
    manager.restart(debate_id, from_round=2)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InterventionRecord:
    """A single operator intervention record."""

    action: str  # "pause" | "resume" | "restart" | "inject_context"
    timestamp: str
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "action": self.action,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "details": self.details,
        }


@dataclass
class InterventionStatus:
    """Current intervention status for a debate."""

    debate_id: str
    state: str  # "running" | "paused" | "completed" | "failed"
    current_round: int
    total_rounds: int
    paused_at: str | None = None
    pause_reason: str | None = None
    interventions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "debate_id": self.debate_id,
            "state": self.state,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "paused_at": self.paused_at,
            "pause_reason": self.pause_reason,
            "interventions": self.interventions,
        }


class _DebateEntry:
    """Internal state for a tracked debate."""

    def __init__(self, debate_id: str, total_rounds: int) -> None:
        self.debate_id = debate_id
        self.total_rounds = total_rounds
        self.current_round: int = 0
        self.state: str = "running"
        self.paused_at: str | None = None
        self.pause_reason: str | None = None
        self.interventions: list[InterventionRecord] = []
        self.injected_contexts: list[str] = []
        self.restart_from_round: int | None = None
        # asyncio.Event: set = running, cleared = paused
        # We create this lazily to avoid requiring an event loop at init time
        self._pause_event: asyncio.Event | None = None

    def get_pause_event(self) -> asyncio.Event:
        """Get or create the asyncio.Event for pause/resume signaling."""
        if self._pause_event is None:
            self._pause_event = asyncio.Event()
            if self.state != "paused":
                self._pause_event.set()
        return self._pause_event


class DebateInterventionManager:
    """Manages operator interventions on running debates.

    Thread-safe: uses a lock for state mutations. The asyncio.Event-based
    pause mechanism is safe across async tasks.
    """

    def __init__(self) -> None:
        self._debates: dict[str, _DebateEntry] = {}
        self._lock = threading.Lock()

    def register(
        self,
        debate_id: str,
        total_rounds: int = 0,
    ) -> None:
        """Register a new debate for operator intervention tracking.

        Args:
            debate_id: Unique debate identifier.
            total_rounds: Expected number of rounds.
        """
        with self._lock:
            if debate_id in self._debates:
                logger.debug(
                    "Debate %s already registered, updating total_rounds", debate_id
                )
                self._debates[debate_id].total_rounds = total_rounds
                return
            self._debates[debate_id] = _DebateEntry(debate_id, total_rounds)
            logger.info(
                "debate_registered debate_id=%s total_rounds=%s",
                debate_id,
                total_rounds,
            )

    def unregister(self, debate_id: str) -> None:
        """Remove a debate from tracking.

        Args:
            debate_id: Debate to remove.
        """
        with self._lock:
            entry = self._debates.pop(debate_id, None)
            if entry:
                logger.info("debate_unregistered debate_id=%s", debate_id)

    def update_round(self, debate_id: str, current_round: int) -> None:
        """Update the current round number for a debate.

        Called by the debate loop at the start of each round.

        Args:
            debate_id: Debate to update.
            current_round: Current round number (1-based).
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry:
                entry.current_round = current_round

    def mark_completed(self, debate_id: str) -> None:
        """Mark a debate as completed.

        Args:
            debate_id: Debate to mark.
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry:
                entry.state = "completed"
                # Ensure any waiters are released
                if entry._pause_event is not None:
                    entry._pause_event.set()

    def mark_failed(self, debate_id: str) -> None:
        """Mark a debate as failed.

        Args:
            debate_id: Debate to mark.
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry:
                entry.state = "failed"
                if entry._pause_event is not None:
                    entry._pause_event.set()

    # ------------------------------------------------------------------
    # Operator Controls
    # ------------------------------------------------------------------

    def pause(self, debate_id: str, reason: str = "") -> bool:
        """Pause a running debate.

        The debate loop should call ``await wait_if_paused(debate_id)``
        between phases to honor this.

        Args:
            debate_id: Debate to pause.
            reason: Human-readable reason for pausing.

        Returns:
            True if the debate was paused, False if not found or wrong state.
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry is None:
                logger.warning("pause_failed: debate %s not found", debate_id)
                return False
            if entry.state != "running":
                logger.warning(
                    "pause_failed: debate %s in state %s", debate_id, entry.state
                )
                return False

            entry.state = "paused"
            entry.paused_at = datetime.now(timezone.utc).isoformat()
            entry.pause_reason = reason or None
            if entry._pause_event is not None:
                entry._pause_event.clear()

            record = InterventionRecord(
                action="pause",
                timestamp=entry.paused_at,
                reason=reason or None,
            )
            entry.interventions.append(record)

            logger.info(
                "debate_paused debate_id=%s reason=%s", debate_id, reason or "(none)"
            )
            return True

    def resume(self, debate_id: str) -> bool:
        """Resume a paused debate.

        Args:
            debate_id: Debate to resume.

        Returns:
            True if the debate was resumed, False if not found or wrong state.
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry is None:
                logger.warning("resume_failed: debate %s not found", debate_id)
                return False
            if entry.state != "paused":
                logger.warning(
                    "resume_failed: debate %s in state %s", debate_id, entry.state
                )
                return False

            entry.state = "running"
            now_str = datetime.now(timezone.utc).isoformat()
            entry.paused_at = None
            entry.pause_reason = None
            if entry._pause_event is not None:
                entry._pause_event.set()

            record = InterventionRecord(
                action="resume",
                timestamp=now_str,
            )
            entry.interventions.append(record)

            logger.info("debate_resumed debate_id=%s", debate_id)
            return True

    def restart(self, debate_id: str, from_round: int = 0) -> bool:
        """Restart a debate from the beginning or a specific round.

        Sets a restart marker that the debate loop can check. The actual
        restart is performed by the orchestrator re-entering the debate
        rounds phase from the specified round.

        Args:
            debate_id: Debate to restart.
            from_round: Round to restart from (0 = beginning).

        Returns:
            True if the restart was scheduled, False if not found or already completed.
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry is None:
                logger.warning("restart_failed: debate %s not found", debate_id)
                return False
            if entry.state == "completed":
                logger.warning("restart_failed: debate %s is completed", debate_id)
                return False

            now_str = datetime.now(timezone.utc).isoformat()
            entry.restart_from_round = max(0, from_round)
            entry.state = "running"
            entry.paused_at = None
            entry.pause_reason = None
            if entry._pause_event is not None:
                entry._pause_event.set()

            record = InterventionRecord(
                action="restart",
                timestamp=now_str,
                details={"from_round": entry.restart_from_round},
            )
            entry.interventions.append(record)

            logger.info(
                "debate_restart_scheduled debate_id=%s from_round=%s",
                debate_id,
                entry.restart_from_round,
            )
            return True

    def inject_context(self, debate_id: str, context: str) -> bool:
        """Inject additional context into a running or paused debate.

        The injected context is stored and can be consumed by the debate
        loop at the next round boundary.

        Args:
            debate_id: Debate to inject into.
            context: Context text to inject.

        Returns:
            True if context was injected, False if debate not found or terminated.
        """
        if not context or not context.strip():
            return False

        with self._lock:
            entry = self._debates.get(debate_id)
            if entry is None:
                logger.warning("inject_context_failed: debate %s not found", debate_id)
                return False
            if entry.state in ("completed", "failed"):
                logger.warning(
                    "inject_context_failed: debate %s in state %s",
                    debate_id,
                    entry.state,
                )
                return False

            now_str = datetime.now(timezone.utc).isoformat()
            entry.injected_contexts.append(context.strip())

            record = InterventionRecord(
                action="inject_context",
                timestamp=now_str,
                details={"context_length": len(context.strip())},
            )
            entry.interventions.append(record)

            logger.info(
                "context_injected debate_id=%s length=%s",
                debate_id,
                len(context.strip()),
            )
            return True

    def get_status(self, debate_id: str) -> InterventionStatus | None:
        """Get the current intervention status for a debate.

        Args:
            debate_id: Debate to query.

        Returns:
            InterventionStatus or None if debate is not tracked.
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry is None:
                return None
            return InterventionStatus(
                debate_id=entry.debate_id,
                state=entry.state,
                current_round=entry.current_round,
                total_rounds=entry.total_rounds,
                paused_at=entry.paused_at,
                pause_reason=entry.pause_reason,
                interventions=[r.to_dict() for r in entry.interventions],
            )

    def list_active(self) -> list[InterventionStatus]:
        """List all active (non-completed, non-failed) debates.

        Returns:
            List of InterventionStatus for debates in running or paused state.
        """
        result: list[InterventionStatus] = []
        with self._lock:
            for entry in self._debates.values():
                if entry.state in ("running", "paused"):
                    result.append(
                        InterventionStatus(
                            debate_id=entry.debate_id,
                            state=entry.state,
                            current_round=entry.current_round,
                            total_rounds=entry.total_rounds,
                            paused_at=entry.paused_at,
                            pause_reason=entry.pause_reason,
                            interventions=[r.to_dict() for r in entry.interventions],
                        )
                    )
        return result

    # ------------------------------------------------------------------
    # Debate Loop Integration
    # ------------------------------------------------------------------

    async def wait_if_paused(self, debate_id: str) -> None:
        """Wait until the debate is no longer paused.

        The debate loop should call this between phases. If the debate
        is paused, this coroutine blocks until ``resume()`` is called.

        Args:
            debate_id: Debate to check.
        """
        entry: _DebateEntry | None = None
        with self._lock:
            entry = self._debates.get(debate_id)

        if entry is None:
            return

        event = entry.get_pause_event()
        await event.wait()

    def consume_restart(self, debate_id: str) -> int | None:
        """Consume and return the pending restart round, if any.

        The debate loop should call this between phases to check if a
        restart has been requested.

        Args:
            debate_id: Debate to check.

        Returns:
            The round to restart from, or None if no restart pending.
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry is None:
                return None
            restart_round = entry.restart_from_round
            entry.restart_from_round = None
            return restart_round

    def consume_injected_contexts(self, debate_id: str) -> list[str]:
        """Consume and return all pending injected contexts.

        The debate loop should call this at round boundaries to pick up
        operator-injected context.

        Args:
            debate_id: Debate to check.

        Returns:
            List of injected context strings (may be empty).
        """
        with self._lock:
            entry = self._debates.get(debate_id)
            if entry is None:
                return []
            contexts = list(entry.injected_contexts)
            entry.injected_contexts.clear()
            return contexts

    def _reset(self) -> None:
        """Clear all tracked debates. For testing only."""
        with self._lock:
            self._debates.clear()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_manager: DebateInterventionManager | None = None
_global_manager_lock = threading.Lock()


def get_operator_manager() -> DebateInterventionManager:
    """Get or create the global DebateInterventionManager singleton.

    Returns:
        The global DebateInterventionManager instance.
    """
    global _global_manager
    if _global_manager is None:
        with _global_manager_lock:
            if _global_manager is None:
                _global_manager = DebateInterventionManager()
    return _global_manager


def _reset_operator_manager() -> None:
    """Reset the global manager singleton. For testing only."""
    global _global_manager
    with _global_manager_lock:
        if _global_manager is not None:
            _global_manager._reset()
        _global_manager = None


__all__ = [
    "DebateInterventionManager",
    "InterventionRecord",
    "InterventionStatus",
    "get_operator_manager",
]
