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
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def from_known_fields(cls: type[_T], data: dict[str, Any]) -> _T:
    """Construct ``cls`` from ``data``, dropping unknown keys instead of crashing.

    Forward-compat for on-disk state/ledger written by a newer schema: an extra
    field must degrade gracefully (skip it), not raise ``TypeError`` and hard-fail
    the whole mission load.
    """
    names = {f.name for f in dataclass_fields(cast(Any, cls))}
    factory = cast(Any, cls)
    return cast(_T, factory(**{k: v for k, v in data.items() if k in names}))


# POSIX-only. Used for the single-writer *owner fence* below — not for serializing
# individual saves (atomic os.replace already prevents torn reads).
fcntl: ModuleType | None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None


class MissionOwnershipError(RuntimeError):
    """Raised when a second writer tries to drive a mission already owned."""


@contextlib.contextmanager
def mission_owner_lock(state_path: str | Path, *, exclusive: bool = True) -> Iterator[None]:
    """Enforce the mission's concurrency contract with a **non-blocking** fence that
    makes orchestrator-mode and swarm-mode mutually exclusive.

    It is a shared/exclusive (reader/writer) lock:

    * ``exclusive=True`` (orchestrator ``run``/``tick``, ``reconcile_from_ledger``):
      takes ``LOCK_EX`` — refused if *anything* else holds the mission (another
      orchestrator, or any live swarm worker). Single writer, enforced.
    * ``exclusive=False`` (swarm ``run_worker``): takes ``LOCK_SH`` — many workers
      coexist (shared), but an orchestrator's ``LOCK_EX`` is refused while any worker
      holds it, and vice versa.

    So two orchestrators, or an orchestrator concurrent with a swarm, fail fast with
    :class:`MissionOwnershipError` instead of double-dispatching a feature — while a
    real multi-worker swarm still runs in parallel. The per-save ``os.replace`` is
    atomic regardless (no torn reads); this fence is what makes *conflicting writers*
    a loud error. If POSIX ``fcntl`` is unavailable, fail closed instead of
    silently running without a fence.
    """
    path = Path(state_path)
    if fcntl is None:  # pragma: no cover - non-POSIX
        raise MissionOwnershipError(
            "mission owner locking requires POSIX fcntl (not available on this platform)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".owner.lock")
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    lf = lock_path.open("a")  # create-if-missing, no truncate; content is irrelevant
    try:
        try:
            fcntl.flock(lf.fileno(), mode | fcntl.LOCK_NB)
        except OSError as exc:
            held_by = "another orchestrator or a live swarm worker"
            raise MissionOwnershipError(
                f"mission {path.name} is already being driven by {held_by}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
    finally:
        lf.close()


class Status:
    """Feature lifecycle states (string constants for JSON-friendliness)."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    ALL = frozenset({PENDING, IN_PROGRESS, COMPLETED, BLOCKED})


def preconditions_met(preconditions: list[str], completed: set[str]) -> bool:
    """Return True iff every precondition is explicitly satisfied.

    Phase A only supports ``feature:<id>`` dependencies. Any other precondition
    token is intentionally fail-closed/unmet until a later validator teaches the
    mission spine how to prove it.
    """
    return all(p.startswith("feature:") and p[8:] in completed for p in preconditions)


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
    metadata: dict[str, Any] = field(default_factory=dict)
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
            if not preconditions_met(feat.preconditions, completed):
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
        feat = self.get(feature_id)
        feat.status = Status.COMPLETED
        for key in (
            "validation_reopened_by",
            "validation_reopened_reason",
            "validation_reopened_ledger_done_invalidated",
        ):
            feat.metadata.pop(key, None)

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
            features=[from_known_fields(Feature, f) for f in data.get("features", [])],
        )

    def save(self, path: str | Path) -> None:
        """Atomically persist to ``path``: tmp-write + fsync + ``os.replace``.

        Safe against the stated threat model (process ``kill -9`` / 402 / crash): the
        atomic rename means a reader never sees a torn or partial file. Concurrent
        *writers* are prevented at a higher level by :func:`mission_owner_lock` (the
        single-writer fence), not by locking each save — locking the write alone
        would not close the load→decide→save race. Power-loss durability would
        additionally require an fsync of the parent directory after the rename — out
        of scope here.
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
