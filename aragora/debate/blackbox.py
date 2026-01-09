"""
Blackbox Protocol - Flight recorder for debate sessions.

Provides crash recovery and debugging capabilities:
- Automatic state snapshots after each turn
- Error logging without crashing
- Session replay data for analysis
- Incremental persistence for resilience

Inspired by nomic loop debate consensus on system stability.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.core import Message

logger = logging.getLogger(__name__)


@dataclass
class BlackboxEvent:
    """A single event recorded by the blackbox."""

    timestamp: float
    event_type: str  # turn, error, consensus, agent_failure, recovery
    component: str   # orchestrator, agent, consensus, etc.
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BlackboxSnapshot:
    """State snapshot at a point in time."""

    turn_id: int
    timestamp: float
    agents_active: list[str]
    agents_failed: list[str]
    consensus_strength: float
    transcript_length: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class BlackboxRecorder:
    """
    Flight recorder for aragora debates.

    Records state incrementally to allow crash recovery and debugging.
    Designed to be lightweight and non-blocking.

    Usage:
        blackbox = BlackboxRecorder(session_id="debate_123")
        blackbox.snapshot_turn(1, {"agents": [...], "transcript": [...]})
        blackbox.log_error("calibration", "Division by zero")
        blackbox.log_agent_failure("claude", "timeout", 90.0)
    """

    def __init__(
        self,
        session_id: str,
        base_path: Optional[Path] = None,
        max_events: int = 10000,
    ):
        """
        Initialize the blackbox recorder.

        Args:
            session_id: Unique identifier for this session
            base_path: Base directory for storage (defaults to .nomic/sessions)
            max_events: Maximum events to keep in memory before flushing
        """
        self.session_id = session_id
        self.base_path = base_path or Path(".nomic/sessions")
        self.session_path = self.base_path / session_id
        self.session_path.mkdir(parents=True, exist_ok=True)

        self.max_events = max_events
        self.events: list[BlackboxEvent] = []
        self.snapshots: list[BlackboxSnapshot] = []
        self.start_time = time.time()

        # Initialize session metadata
        self._write_metadata()
        logger.info(f"blackbox_init session={session_id} path={self.session_path}")

    def _write_metadata(self) -> None:
        """Write session metadata file."""
        metadata = {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(),
            "start_time": self.start_time,
        }
        meta_path = self.session_path / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def record_event(
        self,
        event_type: str,
        component: str,
        data: Optional[dict] = None,
    ) -> BlackboxEvent:
        """
        Record a generic event.

        Args:
            event_type: Type of event (turn, error, consensus, etc.)
            component: Component that generated the event
            data: Additional event data

        Returns:
            The recorded event
        """
        event = BlackboxEvent(
            timestamp=time.time(),
            event_type=event_type,
            component=component,
            data=data or {},
        )
        self.events.append(event)

        # Auto-flush if too many events
        if len(self.events) >= self.max_events:
            self.flush_events()

        return event

    def snapshot_turn(
        self,
        turn_id: int,
        state_data: dict,
    ) -> BlackboxSnapshot:
        """
        Take a snapshot of the current turn state.

        Args:
            turn_id: Turn number
            state_data: Current state to snapshot

        Returns:
            The created snapshot
        """
        snapshot = BlackboxSnapshot(
            turn_id=turn_id,
            timestamp=time.time(),
            agents_active=state_data.get("agents_active", []),
            agents_failed=state_data.get("agents_failed", []),
            consensus_strength=state_data.get("consensus_strength", 0.0),
            transcript_length=state_data.get("transcript_length", 0),
            metadata=state_data.get("metadata", {}),
        )
        self.snapshots.append(snapshot)

        # Write snapshot to disk atomically
        snapshot_path = self.session_path / f"turn_{turn_id:04d}.json"
        try:
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot.to_dict(), f, indent=2)
            logger.debug(f"blackbox_snapshot turn={turn_id}")
        except Exception as e:
            logger.error(f"blackbox_snapshot_failed turn={turn_id} error={e}")

        return snapshot

    def log_error(
        self,
        component: str,
        error: str,
        recoverable: bool = True,
        context: Optional[dict] = None,
    ) -> BlackboxEvent:
        """
        Log an error without crashing.

        Args:
            component: Component that had the error
            error: Error message
            recoverable: Whether the error was recovered from
            context: Additional context

        Returns:
            The recorded event
        """
        event = self.record_event(
            event_type="error",
            component=component,
            data={
                "error": str(error)[:500],  # Truncate long errors
                "recoverable": recoverable,
                "context": context or {},
            },
        )

        # Also write to error log file
        error_log = self.session_path / "errors.log"
        try:
            with open(error_log, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {component}: {error}\n")
        except Exception:
            pass  # Don't fail on logging failure

        logger.warning(f"blackbox_error component={component} error={error[:100]}")
        return event

    def log_agent_failure(
        self,
        agent_name: str,
        failure_type: str,
        duration_seconds: float,
        context: Optional[dict] = None,
    ) -> BlackboxEvent:
        """
        Log an agent failure for analysis.

        Args:
            agent_name: Name of the failed agent
            failure_type: Type of failure (timeout, error, circuit_open)
            duration_seconds: How long until failure
            context: Additional context

        Returns:
            The recorded event
        """
        return self.record_event(
            event_type="agent_failure",
            component=agent_name,
            data={
                "failure_type": failure_type,
                "duration_seconds": duration_seconds,
                "context": context or {},
            },
        )

    def log_recovery(
        self,
        component: str,
        recovery_type: str,
        original_error: str,
        context: Optional[dict] = None,
    ) -> BlackboxEvent:
        """
        Log a successful recovery from an error.

        Args:
            component: Component that recovered
            recovery_type: How recovery was achieved
            original_error: The error that was recovered from
            context: Additional context

        Returns:
            The recorded event
        """
        return self.record_event(
            event_type="recovery",
            component=component,
            data={
                "recovery_type": recovery_type,
                "original_error": str(original_error)[:200],
                "context": context or {},
            },
        )

    def log_consensus(
        self,
        strength: float,
        participating_agents: list[str],
        topic: str,
        result: Optional[str] = None,
    ) -> BlackboxEvent:
        """
        Log a consensus event.

        Args:
            strength: Consensus strength (0.0-1.0)
            participating_agents: Agents that participated
            topic: Topic of consensus
            result: The consensus result

        Returns:
            The recorded event
        """
        return self.record_event(
            event_type="consensus",
            component="orchestrator",
            data={
                "strength": strength,
                "participating_agents": participating_agents,
                "topic": topic[:200],
                "result": result[:500] if result else None,
            },
        )

    def flush_events(self) -> None:
        """Flush events to disk."""
        if not self.events:
            return

        events_path = self.session_path / "events.jsonl"
        try:
            with open(events_path, 'a') as f:
                for event in self.events:
                    f.write(json.dumps(event.to_dict()) + "\n")
            logger.debug(f"blackbox_flush events={len(self.events)}")
            self.events = []
        except Exception as e:
            logger.error(f"blackbox_flush_failed error={e}")

    def get_latest_snapshot(self) -> Optional[BlackboxSnapshot]:
        """Get the most recent snapshot."""
        if self.snapshots:
            return self.snapshots[-1]
        return None

    def get_agent_failure_rate(self, agent_name: str) -> float:
        """Calculate failure rate for an agent."""
        agent_events = [
            e for e in self.events
            if e.component == agent_name
        ]
        if not agent_events:
            return 0.0

        failures = sum(1 for e in agent_events if e.event_type == "agent_failure")
        return failures / len(agent_events)

    def get_session_summary(self) -> dict:
        """Get a summary of the session for debugging."""
        total_errors = sum(1 for e in self.events if e.event_type == "error")
        total_failures = sum(1 for e in self.events if e.event_type == "agent_failure")
        total_recoveries = sum(1 for e in self.events if e.event_type == "recovery")

        return {
            "session_id": self.session_id,
            "duration_seconds": time.time() - self.start_time,
            "total_events": len(self.events),
            "total_snapshots": len(self.snapshots),
            "total_errors": total_errors,
            "total_agent_failures": total_failures,
            "total_recoveries": total_recoveries,
            "recovery_rate": total_recoveries / max(total_errors, 1),
        }

    def close(self) -> None:
        """Close the recorder and flush remaining data."""
        self.flush_events()

        # Write final summary
        summary_path = self.session_path / "summary.json"
        try:
            with open(summary_path, 'w') as f:
                json.dump(self.get_session_summary(), f, indent=2)
        except Exception as e:
            logger.error(f"blackbox_close_failed error={e}")

        logger.info(f"blackbox_close session={self.session_id}")


# Convenience function for getting a session blackbox
_active_recorders: dict[str, BlackboxRecorder] = {}


def get_blackbox(session_id: str) -> BlackboxRecorder:
    """
    Get or create a blackbox recorder for a session.

    Args:
        session_id: Session identifier

    Returns:
        BlackboxRecorder instance
    """
    if session_id not in _active_recorders:
        _active_recorders[session_id] = BlackboxRecorder(session_id)
    return _active_recorders[session_id]


def close_blackbox(session_id: str) -> None:
    """Close and remove a blackbox recorder."""
    if session_id in _active_recorders:
        _active_recorders[session_id].close()
        del _active_recorders[session_id]
