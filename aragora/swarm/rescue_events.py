"""RescueEvent ledger for tracking low-value human interventions.

Every time a human has to copy-paste, type "proceed", rewrite an issue,
approve a permission prompt, or restart a stuck session, that intervention
should be captured as a typed RescueEvent so the system can learn to
absorb repeated rescue patterns autonomously.

The ledger persists to ~/.aragora/rescue_events.jsonl and is designed
to be consumed by the TW-03 rescue-class harvest loop and the RS-11b
RescuePlanner.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_RESCUE_LEDGER_PATH",
    "RescueEvent",
    "RescueEventLedger",
    "RescueEventType",
    "record_rescue",
]

logger = logging.getLogger(__name__)
UTC = timezone.utc

DEFAULT_RESCUE_LEDGER_PATH = Path.home() / ".aragora" / "rescue_events.jsonl"


class RescueEventType(str, Enum):
    """Types of low-value human interventions."""

    FOLLOWUP_PROMPT = "followup_prompt"
    PERMISSION_APPROVAL = "permission_approval"
    ISSUE_REWRITE = "issue_rewrite"
    ISSUE_REQUEUE = "issue_requeue"
    SESSION_RESTART = "session_restart"
    SESSION_KILL = "session_kill"
    PR_SHEPHERD = "pr_shepherd"
    BLOCKED_ESCALATE = "blocked_escalate"
    MANUAL_MERGE = "manual_merge"
    WORKTREE_CLEANUP = "worktree_cleanup"
    COPY_PASTE_RELAY = "copy_paste_relay"
    OTHER = "other"


@dataclass
class RescueEvent:
    """A single recorded human intervention."""

    event_type: str
    reason: str
    actor: str = "founder"
    issue_number: int | None = None
    lane_id: str = ""
    run_id: str = ""
    session_id: str = ""
    pr_number: int | None = None
    evidence: str = ""
    outcome: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class RescueEventLedger:
    """Append-only JSONL ledger for rescue events."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DEFAULT_RESCUE_LEDGER_PATH

    @property
    def path(self) -> Path:
        return self._path

    def record(self, event: RescueEvent) -> None:
        """Append a rescue event to the ledger."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), sort_keys=True))
                f.write("\n")
        except OSError:
            logger.debug("Failed to write rescue event", exc_info=True)

    def recent(self, limit: int = 50) -> list[RescueEvent]:
        """Read the most recent rescue events."""
        if not self._path.exists():
            return []
        events: list[RescueEvent] = []
        try:
            for line in self._path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    data = json.loads(line)
                    events.append(_parse_event(data))
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
        except OSError:
            return []
        return events[-limit:]

    def count_by_type(self) -> dict[str, int]:
        """Count rescue events by type for pattern detection."""
        from collections import Counter

        counts: Counter[str] = Counter()
        events = self.recent(limit=500)
        for event in events:
            counts[event.event_type] += 1
        return dict(counts.most_common())

    def repeated_classes(self, threshold: int = 2) -> list[dict[str, Any]]:
        """Identify rescue event types that appear repeatedly.

        These are candidates for absorption into playbooks or substrate fixes.
        """
        from collections import Counter

        type_reasons: Counter[str] = Counter()
        events = self.recent(limit=500)
        for event in events:
            key = f"{event.event_type}:{event.reason[:60]}"
            type_reasons[key] += 1
        return [
            {"class": key, "count": count}
            for key, count in type_reasons.most_common()
            if count >= threshold
        ]


def _parse_event(data: dict[str, Any]) -> RescueEvent:
    """Parse a rescue event from a JSONL row."""
    fields = RescueEvent.__dataclass_fields__.keys()
    return RescueEvent(**{k: v for k, v in data.items() if k in fields})


def record_rescue(
    event_type: str,
    reason: str,
    *,
    actor: str = "founder",
    issue_number: int | None = None,
    lane_id: str = "",
    session_id: str = "",
    pr_number: int | None = None,
    evidence: str = "",
    outcome: str = "",
    ledger_path: Path | None = None,
) -> None:
    """Convenience function to record a rescue event."""
    ledger = RescueEventLedger(path=ledger_path)
    ledger.record(
        RescueEvent(
            event_type=event_type,
            reason=reason,
            actor=actor,
            issue_number=issue_number,
            lane_id=lane_id,
            session_id=session_id,
            pr_number=pr_number,
            evidence=evidence,
            outcome=outcome,
        )
    )
