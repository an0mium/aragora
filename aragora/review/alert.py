"""Edge-triggered alerter for the proof-loop write-side daemons.

Designed to be run periodically from launchd (or any scheduler) and produce
one alert event when the set of stale/missing surfaces changes, instead of
spamming events every tick.

Algorithm:
    1. Run :func:`aragora.review.health.gather_health` to get the current
       observation.
    2. Extract the set of "alerting" surfaces — those whose status is
       ``stale`` or ``missing``. ``aging`` is intentionally not alerting
       because it is recoverable and frequently transient.
    3. Compare to the previously persisted state. If the set changed (new
       stale surface, or full recovery), write an event JSON to disk.
    4. Always persist the latest state and timestamps so an operator can
       see the alerter itself is alive even when nothing fired.

This is read-only with respect to the queue itself: it only writes under
``.aragora/proof-loop-alerts/`` and never mutates queue receipts, briefs,
or any other proof-loop artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.review.health import (
    STATUS_MISSING,
    STATUS_STALE,
    HealthReport,
    SurfaceCheck,
    gather_health,
)

UTC = timezone.utc

DEFAULT_ALERTS_REL = Path(".aragora") / "proof-loop-alerts"
STATE_FILENAME = "state.json"
EVENTS_SUBDIR = "events"

EVENT_FILENAME_PREFIX = "event-"
EVENT_FILENAME_SUFFIX = ".json"

# Conservative default cap on the number of event files retained under
# ``events/``. The alerter is designed for periodic launchd execution, so
# without a cap a long-running heartbeat schedule would accumulate event
# files indefinitely. 200 covers ~2 days of 15-minute heartbeats or
# hundreds of real state-transition events; either way the operator-
# relevant tail of recent events is preserved while bounding disk use.
# A value <= 0 disables pruning (caller can opt out explicitly).
DEFAULT_MAX_EVENTS = 200

EVENT_KIND_OPENED = "alert_opened"
EVENT_KIND_CHANGED = "alert_changed"
EVENT_KIND_RECOVERED = "alert_recovered"
EVENT_KIND_HEARTBEAT = "heartbeat"

# Statuses that constitute an active alert. ``aging`` is deliberately
# excluded — it is the warning band, not the alarm band.
ALERTING_STATUSES: frozenset[str] = frozenset({STATUS_STALE, STATUS_MISSING})


@dataclass(frozen=True)
class AlertState:
    """Persisted state between alerter runs."""

    alerting_surfaces: list[str] = field(default_factory=list)
    last_event_at: datetime | None = None
    last_run_at: datetime | None = None
    last_event_kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "alerting_surfaces": list(self.alerting_surfaces),
            "last_event_at": (
                self.last_event_at.isoformat() if self.last_event_at is not None else None
            ),
            "last_run_at": (self.last_run_at.isoformat() if self.last_run_at is not None else None),
            "last_event_kind": self.last_event_kind,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlertState:
        surfaces_raw = data.get("alerting_surfaces") or []
        if not isinstance(surfaces_raw, list):
            surfaces_raw = []
        surfaces = [str(s) for s in surfaces_raw if isinstance(s, str)]
        return cls(
            alerting_surfaces=surfaces,
            last_event_at=_parse_dt(data.get("last_event_at")),
            last_run_at=_parse_dt(data.get("last_run_at")),
            last_event_kind=(
                str(data["last_event_kind"]) if data.get("last_event_kind") is not None else None
            ),
        )


@dataclass(frozen=True)
class AlertEvent:
    """A single emitted alert event."""

    kind: str
    generated_at: datetime
    previous_alerting: list[str]
    current_alerting: list[str]
    surfaces: list[dict[str, object]]
    overall_status: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "generated_at": self.generated_at.isoformat(),
            "previous_alerting": list(self.previous_alerting),
            "current_alerting": list(self.current_alerting),
            "overall_status": self.overall_status,
            "surfaces": list(self.surfaces),
        }


@dataclass(frozen=True)
class AlertResult:
    """Result of one alerter run."""

    state: AlertState
    event: AlertEvent | None
    report: HealthReport
    state_path: Path
    event_path: Path | None


def _parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        text = value.rstrip("Z")
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def alerting_surface_names(report: HealthReport) -> list[str]:
    """Return the sorted list of surface names whose status is stale/missing."""
    return sorted(s.name for s in report.surfaces if s.status in ALERTING_STATUSES)


def _surface_payload(surface: SurfaceCheck) -> dict[str, object]:
    """Pare a SurfaceCheck down to the fields useful in an alert event."""
    return {
        "name": surface.name,
        "status": surface.status,
        "age_hours": (round(surface.age_hours, 2) if surface.age_hours is not None else None),
        "count": surface.count,
        "path": surface.path,
        "detail": surface.detail,
    }


def determine_event_kind(
    previous: list[str],
    current: list[str],
    *,
    emit_heartbeat: bool = False,
) -> str | None:
    """Decide whether and what kind of event to emit.

    Returns ``None`` when the state did not change (and a heartbeat was not
    requested). The caller is responsible for persisting state regardless.
    """
    prev_set = set(previous)
    cur_set = set(current)
    if not prev_set and not cur_set:
        return EVENT_KIND_HEARTBEAT if emit_heartbeat else None
    if not prev_set and cur_set:
        return EVENT_KIND_OPENED
    if prev_set and not cur_set:
        return EVENT_KIND_RECOVERED
    if prev_set != cur_set:
        return EVENT_KIND_CHANGED
    return EVENT_KIND_HEARTBEAT if emit_heartbeat else None


def load_state(path: Path) -> AlertState:
    """Read persisted state. Missing/corrupt files yield a fresh state."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return AlertState()
    except OSError:
        return AlertState()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return AlertState()
    if not isinstance(data, dict):
        return AlertState()
    return AlertState.from_dict(data)


def save_state(state: AlertState, path: Path) -> None:
    """Atomically write state.json so a partial write can never corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = path.parent
    fd, tmp_path = tempfile.mkstemp(prefix=".state-", suffix=".json", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state.to_dict(), fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _safe_kind_slug(kind: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in kind) or "event"


def write_event(event: AlertEvent, events_dir: Path) -> Path:
    """Atomically write the event JSON. Returns the path that was written.

    Uses the same tempfile + ``os.replace`` pattern as :func:`save_state`:
    the JSON is first written to a hidden temp file in the destination
    directory, then renamed into place. If the process is interrupted
    mid-write, the final file either does not exist or contains the
    complete event — there is no observable partial state. A cleanup
    on failure removes the temp file so the events directory does not
    accumulate stray ``.event-*`` fragments.
    """
    events_dir.mkdir(parents=True, exist_ok=True)
    ts = event.generated_at.strftime("%Y%m%dT%H%M%SZ")
    name = f"event-{ts}-{_safe_kind_slug(event.kind)}.json"
    path = events_dir / name
    suffix = 1
    while path.exists():
        path = events_dir / f"event-{ts}-{_safe_kind_slug(event.kind)}-{suffix}.json"
        suffix += 1
    payload = json.dumps(event.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, tmp_path_str = tempfile.mkstemp(prefix=".event-", suffix=".json", dir=events_dir)
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_path, path)
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    return path


def prune_event_files(
    events_dir: Path,
    *,
    max_count: int = DEFAULT_MAX_EVENTS,
) -> list[Path]:
    """Remove oldest ``event-*.json`` files until at most ``max_count`` remain.

    Conservative and local-only:

    * Only files whose name starts with ``event-`` and ends with ``.json``
      are considered; any unrelated file in ``events_dir`` (e.g. an
      operator-placed ``README.md`` or backup) is never touched.
    * Order is by file mtime (oldest first), which is robust against the
      collision-suffix scheme in :func:`write_event` (where two events at
      the same second produce names that do not sort chronologically).
    * A non-existent ``events_dir`` is treated as empty (returns ``[]``).
    * ``max_count`` <= 0 disables pruning (returns ``[]``); useful for
      callers that want to opt out and retain everything.
    * Individual unlink failures are tolerated — pruning is best-effort
      housekeeping and must not break the alerter run.

    Returns the list of paths actually removed (for testability).
    """
    if max_count <= 0:
        return []
    if not events_dir.exists() or not events_dir.is_dir():
        return []
    event_files: list[Path] = []
    for entry in events_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if not name.startswith(EVENT_FILENAME_PREFIX) or not name.endswith(EVENT_FILENAME_SUFFIX):
            continue
        event_files.append(entry)
    if len(event_files) <= max_count:
        return []
    event_files.sort(key=lambda p: p.stat().st_mtime)
    excess = len(event_files) - max_count
    removed: list[Path] = []
    for path in event_files[:excess]:
        try:
            path.unlink()
        except OSError:
            continue
        removed.append(path)
    return removed


def evaluate(
    report: HealthReport,
    *,
    state_dir: Path,
    emit_heartbeat: bool = False,
    now: datetime | None = None,
) -> AlertResult:
    """Pure decision step — no I/O. Run by the CLI handler after gathering health."""
    now = now if now is not None else datetime.now(tz=UTC)
    state_path = state_dir / STATE_FILENAME
    prev_state = load_state(state_path)
    current_alerting = alerting_surface_names(report)
    kind = determine_event_kind(
        prev_state.alerting_surfaces,
        current_alerting,
        emit_heartbeat=emit_heartbeat,
    )
    event: AlertEvent | None = None
    if kind is not None:
        relevant = [
            _surface_payload(s)
            for s in report.surfaces
            if (s.name in set(current_alerting) | set(prev_state.alerting_surfaces))
        ]
        if not relevant and emit_heartbeat:
            relevant = [_surface_payload(s) for s in report.surfaces]
        event = AlertEvent(
            kind=kind,
            generated_at=now,
            previous_alerting=list(prev_state.alerting_surfaces),
            current_alerting=list(current_alerting),
            surfaces=relevant,
            overall_status=report.overall_status,
        )
    new_state = AlertState(
        alerting_surfaces=current_alerting,
        last_event_at=(event.generated_at if event is not None else prev_state.last_event_at),
        last_run_at=now,
        last_event_kind=(event.kind if event is not None else prev_state.last_event_kind),
    )
    return AlertResult(
        state=new_state,
        event=event,
        report=report,
        state_path=state_path,
        event_path=None,
    )


def run_alert(
    *,
    state_dir: Path,
    emit_heartbeat: bool = False,
    repo_root: Path | None = None,
    review_queue_root: Path | None = None,
    overnight_root: Path | None = None,
    automation_receipts_root: Path | None = None,
    max_events: int = DEFAULT_MAX_EVENTS,
) -> AlertResult:
    """End-to-end run: gather health, evaluate, persist state and any event.

    This is the function the CLI handler invokes. It performs I/O.

    ``max_events`` bounds the size of the events subdirectory; old files
    beyond the cap are pruned after each write. Pass ``max_events <= 0``
    to disable pruning entirely.
    """
    report = gather_health(
        repo_root=repo_root,
        review_queue_root=review_queue_root,
        overnight_root=overnight_root,
        automation_receipts_root=automation_receipts_root,
    )
    decision = evaluate(report, state_dir=state_dir, emit_heartbeat=emit_heartbeat)
    event_path: Path | None = None
    if decision.event is not None:
        event_path = write_event(decision.event, state_dir / EVENTS_SUBDIR)
        prune_event_files(state_dir / EVENTS_SUBDIR, max_count=max_events)
    save_state(decision.state, decision.state_path)
    return AlertResult(
        state=decision.state,
        event=decision.event,
        report=decision.report,
        state_path=decision.state_path,
        event_path=event_path,
    )


def resolve_state_dir(repo_root: Path) -> Path:
    return repo_root / DEFAULT_ALERTS_REL
