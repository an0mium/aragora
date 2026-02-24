"""
Debate Intervention Manager.

Provides real-time human intervention capabilities for live debates,
including pause/resume, nudges, challenges, and evidence injection.

Thread-safe state management ensures correct behavior under concurrent access.

Usage:
    from aragora.debate.intervention import InterventionManager

    manager = InterventionManager(debate_id="abc-123")
    manager.pause(user_id="admin-1")
    manager.nudge("Consider the economic implications", user_id="admin-1")
    manager.resume(user_id="admin-1")
    log = manager.get_log()
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of interventions that can be applied to a debate."""

    PAUSE = "pause"
    RESUME = "resume"
    NUDGE = "nudge"
    CHALLENGE = "challenge"
    INJECT_EVIDENCE = "inject_evidence"


class DebateInterventionState(Enum):
    """Current state of a debate with respect to interventions."""

    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class InterventionEntry:
    """A single intervention record."""

    intervention_type: InterventionType
    timestamp: float
    user_id: str | None = None
    message: str | None = None
    target_agent: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "type": self.intervention_type.value,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "message": self.message,
            "target_agent": self.target_agent,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class InterventionLog:
    """Full log of all interventions for a debate."""

    debate_id: str
    entries: list[InterventionEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "debate_id": self.debate_id,
            "entry_count": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }


class InterventionManager:
    """Manages real-time human interventions for a live debate.

    Thread-safe: all public methods acquire the internal lock before mutating state.

    Args:
        debate_id: The debate this manager controls.
        emitter: Optional SyncEventEmitter for broadcasting WebSocket events.
    """

    def __init__(
        self,
        debate_id: str,
        emitter: Any | None = None,
    ) -> None:
        self._debate_id = debate_id
        self._state = DebateInterventionState.RUNNING
        self._log = InterventionLog(debate_id=debate_id)
        self._lock = threading.Lock()
        self._emitter = emitter

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def debate_id(self) -> str:
        """Return the debate ID managed by this instance."""
        return self._debate_id

    @property
    def state(self) -> DebateInterventionState:
        """Return the current intervention state."""
        with self._lock:
            return self._state

    @property
    def is_paused(self) -> bool:
        """Return True if the debate is currently paused."""
        with self._lock:
            return self._state == DebateInterventionState.PAUSED

    @property
    def is_running(self) -> bool:
        """Return True if the debate is currently running."""
        with self._lock:
            return self._state == DebateInterventionState.RUNNING

    @property
    def is_completed(self) -> bool:
        """Return True if the debate is completed."""
        with self._lock:
            return self._state == DebateInterventionState.COMPLETED

    # ------------------------------------------------------------------
    # Intervention Operations
    # ------------------------------------------------------------------

    def pause(self, user_id: str | None = None) -> InterventionEntry:
        """Pause a running debate.

        Args:
            user_id: ID of the user initiating the pause.

        Returns:
            The recorded intervention entry.

        Raises:
            ValueError: If the debate is not in a running state.
        """
        with self._lock:
            if self._state != DebateInterventionState.RUNNING:
                raise ValueError(
                    f"Cannot pause debate {self._debate_id}: "
                    f"current state is {self._state.value}"
                )
            self._state = DebateInterventionState.PAUSED
            entry = self._record(InterventionType.PAUSE, user_id=user_id)
            self._emit_event("debate_paused", entry)
            logger.info("Debate %s paused by %s", self._debate_id, user_id or "unknown")
            return entry

    def resume(self, user_id: str | None = None) -> InterventionEntry:
        """Resume a paused debate.

        Args:
            user_id: ID of the user initiating the resume.

        Returns:
            The recorded intervention entry.

        Raises:
            ValueError: If the debate is not in a paused state.
        """
        with self._lock:
            if self._state != DebateInterventionState.PAUSED:
                raise ValueError(
                    f"Cannot resume debate {self._debate_id}: "
                    f"current state is {self._state.value}"
                )
            self._state = DebateInterventionState.RUNNING
            entry = self._record(InterventionType.RESUME, user_id=user_id)
            self._emit_event("debate_resumed", entry)
            logger.info("Debate %s resumed by %s", self._debate_id, user_id or "unknown")
            return entry

    def nudge(
        self,
        message: str,
        user_id: str | None = None,
        target_agent: str | None = None,
    ) -> InterventionEntry:
        """Send a nudge/hint to the debate.

        Args:
            message: The nudge message.
            user_id: ID of the user sending the nudge.
            target_agent: Optional specific agent to nudge.

        Returns:
            The recorded intervention entry.

        Raises:
            ValueError: If the debate is completed or message is empty.
        """
        if not message or not message.strip():
            raise ValueError("Nudge message cannot be empty")

        with self._lock:
            if self._state == DebateInterventionState.COMPLETED:
                raise ValueError(
                    f"Cannot nudge completed debate {self._debate_id}"
                )
            entry = self._record(
                InterventionType.NUDGE,
                user_id=user_id,
                message=message.strip(),
                target_agent=target_agent,
            )
            self._emit_event("debate_nudge", entry)
            logger.info(
                "Nudge sent to debate %s%s by %s",
                self._debate_id,
                f" (agent: {target_agent})" if target_agent else "",
                user_id or "unknown",
            )
            return entry

    def challenge(
        self,
        challenge_text: str,
        user_id: str | None = None,
    ) -> InterventionEntry:
        """Inject a challenge/counterargument into the debate.

        Args:
            challenge_text: The challenge to inject.
            user_id: ID of the user injecting the challenge.

        Returns:
            The recorded intervention entry.

        Raises:
            ValueError: If the debate is completed or challenge is empty.
        """
        if not challenge_text or not challenge_text.strip():
            raise ValueError("Challenge text cannot be empty")

        with self._lock:
            if self._state == DebateInterventionState.COMPLETED:
                raise ValueError(
                    f"Cannot challenge completed debate {self._debate_id}"
                )
            entry = self._record(
                InterventionType.CHALLENGE,
                user_id=user_id,
                message=challenge_text.strip(),
            )
            self._emit_event("debate_challenge", entry)
            logger.info(
                "Challenge injected into debate %s by %s",
                self._debate_id,
                user_id or "unknown",
            )
            return entry

    def inject_evidence(
        self,
        evidence: str,
        source: str | None = None,
        user_id: str | None = None,
    ) -> InterventionEntry:
        """Inject evidence into the debate.

        Args:
            evidence: The evidence text to inject.
            source: The source/citation for the evidence.
            user_id: ID of the user injecting the evidence.

        Returns:
            The recorded intervention entry.

        Raises:
            ValueError: If the debate is completed or evidence is empty.
        """
        if not evidence or not evidence.strip():
            raise ValueError("Evidence text cannot be empty")

        with self._lock:
            if self._state == DebateInterventionState.COMPLETED:
                raise ValueError(
                    f"Cannot inject evidence into completed debate {self._debate_id}"
                )
            entry = self._record(
                InterventionType.INJECT_EVIDENCE,
                user_id=user_id,
                message=evidence.strip(),
                source=source,
            )
            self._emit_event("debate_evidence_injected", entry)
            logger.info(
                "Evidence injected into debate %s by %s (source: %s)",
                self._debate_id,
                user_id or "unknown",
                source or "unspecified",
            )
            return entry

    def mark_completed(self) -> None:
        """Mark the debate as completed, preventing further interventions."""
        with self._lock:
            self._state = DebateInterventionState.COMPLETED
            logger.info("Debate %s marked as completed for interventions", self._debate_id)

    def get_log(self) -> InterventionLog:
        """Return the full intervention log.

        Returns:
            A copy of the InterventionLog with all entries.
        """
        with self._lock:
            return InterventionLog(
                debate_id=self._log.debate_id,
                entries=list(self._log.entries),
            )

    def get_state_dict(self) -> dict[str, Any]:
        """Return a dict summarising the current intervention state."""
        with self._lock:
            return {
                "debate_id": self._debate_id,
                "state": self._state.value,
                "intervention_count": len(self._log.entries),
            }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _record(
        self,
        intervention_type: InterventionType,
        user_id: str | None = None,
        message: str | None = None,
        target_agent: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InterventionEntry:
        """Record an intervention entry (caller must hold _lock)."""
        entry = InterventionEntry(
            intervention_type=intervention_type,
            timestamp=time.time(),
            user_id=user_id,
            message=message,
            target_agent=target_agent,
            source=source,
            metadata=metadata or {},
        )
        self._log.entries.append(entry)
        return entry

    def _emit_event(self, event_name: str, entry: InterventionEntry) -> None:
        """Emit a WebSocket event for the intervention (caller must hold _lock)."""
        if self._emitter is None:
            return

        try:
            from aragora.server.stream.events import StreamEvent, StreamEventType

            self._emitter.emit(
                StreamEvent(
                    type=StreamEventType.BREAKPOINT,
                    data={
                        "event": event_name,
                        "debate_id": self._debate_id,
                        "intervention": entry.to_dict(),
                        "state": self._state.value,
                    },
                    loop_id=self._debate_id,
                )
            )
        except (ImportError, AttributeError, TypeError, RuntimeError) as exc:
            logger.debug("Failed to emit intervention event: %s", exc)


# ---------------------------------------------------------------------------
# Module-level registry for active intervention managers
# ---------------------------------------------------------------------------

_managers: dict[str, InterventionManager] = {}
_managers_lock = threading.Lock()


def get_intervention_manager(
    debate_id: str,
    emitter: Any | None = None,
    create: bool = True,
) -> InterventionManager | None:
    """Get or create the InterventionManager for a debate.

    Args:
        debate_id: The debate to look up.
        emitter: Optional emitter passed to a newly created manager.
        create: If True (default), create a new manager when one does not exist.

    Returns:
        The InterventionManager, or None if *create* is False and none exists.
    """
    with _managers_lock:
        manager = _managers.get(debate_id)
        if manager is not None:
            return manager
        if not create:
            return None
        manager = InterventionManager(debate_id=debate_id, emitter=emitter)
        _managers[debate_id] = manager
        return manager


def remove_intervention_manager(debate_id: str) -> InterventionManager | None:
    """Remove and return the InterventionManager for a debate.

    Args:
        debate_id: The debate whose manager should be removed.

    Returns:
        The removed manager, or None if not found.
    """
    with _managers_lock:
        return _managers.pop(debate_id, None)


def list_intervention_managers() -> dict[str, InterventionManager]:
    """Return a copy of all active intervention managers."""
    with _managers_lock:
        return dict(_managers)


def _reset_managers() -> None:
    """Reset the global manager registry (for testing)."""
    with _managers_lock:
        _managers.clear()


__all__ = [
    "InterventionType",
    "DebateInterventionState",
    "InterventionEntry",
    "InterventionLog",
    "InterventionManager",
    "get_intervention_manager",
    "remove_intervention_manager",
    "list_intervention_managers",
]
