"""MissionState — the single on-disk source of truth for a mission.

Mirrors Factory's proven ``features.json`` schema. The whole survivability story
rests on two invariants:

1. Every mutation is followed by an **atomic** ``save()`` (tmp-write + ``os.replace``),
   so a crash mid-write can never corrupt the file.
2. The orchestrator **never** holds state across ticks — it reloads from disk each
   time. This module therefore carries no runtime/process state of its own.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class Status:
    """Feature lifecycle states (string constants for JSON-friendliness)."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    ALL = frozenset({PENDING, IN_PROGRESS, COMPLETED, BLOCKED})


@dataclass
class Feature:
    """One unit of mission work — the atom the orchestrator dispatches."""

    id: str
    description: str
    milestone: str
    skill: str = "worker"
    status: str = Status.PENDING
    preconditions: list[str] = field(default_factory=list)
    expected_behavior: list[str] = field(default_factory=list)
    fulfills: list[str] = field(default_factory=list)  # assertion ids (Phase B)
    worker_session_ids: list[str] = field(default_factory=list)
    retry_count: int = 0  # dispatches that RETURNED failure (bounds the retry loop)
    crash_count: int = 0  # consecutive dispatches that did NOT return (raise/crash)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in Status.ALL:
            raise ValueError(f"invalid status {self.status!r} for feature {self.id!r}")


@dataclass
class MissionState:
    """The full mission: goal + ordered milestones + ordered features."""

    mission_id: str
    goal: str
    milestones: list[str] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)

    # ---- queue / advance API -------------------------------------------------

    def next_pending(self) -> Feature | None:
        """First feature whose status is PENDING and whose preconditions are met.

        Ordering is array order (milestones are encoded by position, exactly like
        Factory's features.json), so callers control sequencing by list order.
        """
        completed = {f.id for f in self.features if f.status == Status.COMPLETED}
        for feat in self.features:
            if feat.status != Status.PENDING:
                continue
            unmet = [
                p for p in feat.preconditions if p.startswith("feature:") and p[8:] not in completed
            ]
            if unmet:
                continue
            return feat
        return None

    def get(self, feature_id: str) -> Feature:
        for feat in self.features:
            if feat.id == feature_id:
                return feat
        raise KeyError(f"no feature {feature_id!r} in mission {self.mission_id!r}")

    def mark_in_progress(self, feature_id: str, session_id: str | None = None) -> None:
        feat = self.get(feature_id)
        feat.status = Status.IN_PROGRESS
        if session_id and session_id not in feat.worker_session_ids:
            feat.worker_session_ids.append(session_id)

    def mark_completed(self, feature_id: str) -> None:
        self.get(feature_id).status = Status.COMPLETED

    def mark_blocked(self, feature_id: str, reason: str = "") -> None:
        feat = self.get(feature_id)
        feat.status = Status.BLOCKED
        if reason:
            feat.notes = (feat.notes + "\n" if feat.notes else "") + f"BLOCKED: {reason}"

    def reclaim_in_progress(self) -> list[str]:
        """Reset orphaned IN_PROGRESS features (from a dead worker) to PENDING.

        Phase-A resume policy: a feature whose worker died mid-run is simply
        retried from scratch (dispatch must be idempotent). A later phase can
        instead resume the worker session via ``worker_session_ids``.
        Returns the ids that were reclaimed.
        """
        reclaimed: list[str] = []
        for feat in self.features:
            if feat.status == Status.IN_PROGRESS:
                feat.status = Status.PENDING
                reclaimed.append(feat.id)
        return reclaimed

    def insert_feature(self, feature: Feature, before: str | None = None) -> None:
        """Insert a follow-up feature (handoff triage extends the queue this way)."""
        if any(f.id == feature.id for f in self.features):
            raise ValueError(f"duplicate feature id {feature.id!r}")
        if before is None:
            self.features.append(feature)
            return
        idx = next((i for i, f in enumerate(self.features) if f.id == before), len(self.features))
        self.features.insert(idx, feature)

    def milestone_complete(self, milestone: str) -> bool:
        feats = [f for f in self.features if f.milestone == milestone]
        return bool(feats) and all(f.status == Status.COMPLETED for f in feats)

    def progress(self) -> tuple[int, int]:
        done = sum(1 for f in self.features if f.status == Status.COMPLETED)
        return done, len(self.features)

    # ---- persistence ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "goal": self.goal,
            "milestones": list(self.milestones),
            "features": [asdict(f) for f in self.features],
        }

    @classmethod
    def from_dict(cls, data: dict) -> MissionState:
        return cls(
            mission_id=data["mission_id"],
            goal=data["goal"],
            milestones=list(data.get("milestones", [])),
            features=[Feature(**f) for f in data.get("features", [])],
        )

    def save(self, path: str | Path) -> None:
        """Atomically persist to ``path``: tmp-write + fsync + ``os.replace``.

        Safe against the stated threat model (process ``kill -9`` / 402 / crash).
        Power-loss durability would additionally require an fsync of the parent
        directory after the rename — out of scope here.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    @classmethod
    def load(cls, path: str | Path) -> MissionState:
        with Path(path).open(encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
