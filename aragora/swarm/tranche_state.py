from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TRANCHE_STATUS_PLANNED = "planned"
TRANCHE_STATUS_PREPARING = "preparing"
TRANCHE_STATUS_RUNNING = "running"
TRANCHE_STATUS_REVIEWING = "reviewing"
TRANCHE_STATUS_INTEGRATING = "integrating"
TRANCHE_STATUS_COMPLETED = "completed"
TRANCHE_STATUS_NEEDS_HUMAN = "needs_human"
TRANCHE_STATUS_ABORTED = "aborted"

LANE_STATUS_PENDING = "pending"
LANE_STATUS_PREPARING = "preparing"
LANE_STATUS_DISPATCHED = "dispatched"
LANE_STATUS_RUNNING = "running"
LANE_STATUS_COMPLETED = "completed"
LANE_STATUS_REVIEWING = "reviewing"
LANE_STATUS_REVIEW_PASSED = "review_passed"
LANE_STATUS_REVIEW_FAILED = "review_failed"
LANE_STATUS_RETRYING = "retrying"
LANE_STATUS_WAITING_FOR_PR = "waiting_for_pr"
LANE_STATUS_WAITING_FOR_MERGE = "waiting_for_merge"
LANE_STATUS_NEEDS_HUMAN = "needs_human"
LANE_STATUS_ABORTED = "aborted"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = _optional_text(value)
    if not text:
        return _utcnow()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return _utcnow()


@dataclass(slots=True)
class LaneRunState:
    lane_id: str
    status: str
    run_id: str | None = None
    receipt_id: str | None = None
    lease_id: str | None = None
    worktree_path: str | None = None
    pr_url: str | None = None
    retry_count: int = 0
    last_updated: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "status": self.status,
            "run_id": self.run_id,
            "receipt_id": self.receipt_id,
            "lease_id": self.lease_id,
            "worktree_path": self.worktree_path,
            "pr_url": self.pr_url,
            "retry_count": self.retry_count,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LaneRunState:
        data = data or {}
        return cls(
            lane_id=str(data.get("lane_id", "")).strip(),
            status=str(data.get("status", LANE_STATUS_PENDING)).strip() or LANE_STATUS_PENDING,
            run_id=_optional_text(data.get("run_id")),
            receipt_id=_optional_text(data.get("receipt_id")),
            lease_id=_optional_text(data.get("lease_id")),
            worktree_path=_optional_text(data.get("worktree_path")),
            pr_url=_optional_text(data.get("pr_url")),
            retry_count=max(0, int(data.get("retry_count", 0) or 0)),
            last_updated=_coerce_datetime(data.get("last_updated")),
        )


@dataclass(slots=True)
class TrancheRunState:
    manifest_id: str
    status: str
    autonomy_mode: str
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    lane_states: dict[str, LaneRunState] = field(default_factory=dict)
    driver_session: str | None = None
    driver_heartbeat: datetime | None = None
    session_history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "status": self.status,
            "autonomy_mode": self.autonomy_mode,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "lane_states": {lane_id: lane.to_dict() for lane_id, lane in self.lane_states.items()},
            "driver_session": self.driver_session,
            "driver_heartbeat": (
                self.driver_heartbeat.isoformat() if self.driver_heartbeat else None
            ),
            "session_history": [dict(item) for item in self.session_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrancheRunState:
        data = data or {}
        lane_states_raw = data.get("lane_states") or {}
        return cls(
            manifest_id=str(data.get("manifest_id", "")).strip(),
            status=str(data.get("status", TRANCHE_STATUS_PLANNED)).strip()
            or TRANCHE_STATUS_PLANNED,
            autonomy_mode=str(data.get("autonomy_mode", "adaptive")).strip() or "adaptive",
            created_at=_coerce_datetime(data.get("created_at")),
            updated_at=_coerce_datetime(data.get("updated_at")),
            lane_states={
                str(lane_id): LaneRunState.from_dict(lane_data)
                for lane_id, lane_data in lane_states_raw.items()
                if str(lane_id).strip() and isinstance(lane_data, dict)
            },
            driver_session=_optional_text(data.get("driver_session")),
            driver_heartbeat=(
                _coerce_datetime(data.get("driver_heartbeat"))
                if _optional_text(data.get("driver_heartbeat"))
                else None
            ),
            session_history=[
                dict(item) for item in data.get("session_history", []) if isinstance(item, dict)
            ],
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        try:
            import yaml

            text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=False)
        except ImportError:
            text = json.dumps(payload, indent=2, sort_keys=True)
        target.write_text(text)

    @classmethod
    def load(cls, path: str | Path) -> TrancheRunState:
        target = Path(path)
        text = target.read_text()
        try:
            import yaml

            data = yaml.safe_load(text) or {}
        except ImportError:
            data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Tranche run state must deserialize to an object.")
        return cls.from_dict(data)
