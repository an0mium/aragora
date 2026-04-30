"""Capability checkpoints for the booster-rocket thesis (AGT-06 / #6067).

Five graduation gates from substrate to surface; see
``docs/plans/AGENT_CIVILIZATION_SUBSTRATE.md`` §5.

Flag: ``ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED`` (default off).
Dataclasses are always constructable; :meth:`CheckpointRegistry.record` is gated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def capability_checkpoints_enabled() -> bool:
    return os.environ.get("ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED", "").lower() in _TRUTHY


class CheckpointCode(str, Enum):
    CP1 = "CP-1"
    CP2 = "CP-2"
    CP3 = "CP-3"
    CP4 = "CP-4"
    CP5 = "CP-5"


class CheckpointStatus(str, Enum):
    PENDING = "pending"
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CapabilityCheckpoint:
    code: CheckpointCode
    title: str
    window_weeks: int
    depends_on: CheckpointCode | None
    pass_condition: str
    action_if_not_met: str


@dataclass
class CheckpointRecord:
    checkpoint_code: CheckpointCode
    status: CheckpointStatus
    evaluated_at: str
    evaluator: str
    evidence: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def create(
        cls,
        *,
        checkpoint_code: CheckpointCode,
        status: CheckpointStatus,
        evaluator: str,
        evidence: dict[str, Any] | None = None,
        notes: str = "",
        evaluated_at: str | None = None,
    ) -> "CheckpointRecord":
        return cls(
            checkpoint_code=checkpoint_code,
            status=status,
            evaluator=evaluator,
            evidence=dict(evidence or {}),
            notes=notes,
            evaluated_at=evaluated_at or datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_code": self.checkpoint_code.value,
            "status": self.status.value,
            "evaluated_at": self.evaluated_at,
            "evaluator": self.evaluator,
            "evidence": self.evidence,
            "notes": self.notes,
        }


class CheckpointRegistryError(RuntimeError):
    pass


class CheckpointRegistry:
    """In-memory registry; reads always allowed, writes are flag-gated."""

    def __init__(self, checkpoints: list[CapabilityCheckpoint]) -> None:
        if not checkpoints:
            raise CheckpointRegistryError("checkpoints must be non-empty")
        self._cps: dict[CheckpointCode, CapabilityCheckpoint] = {cp.code: cp for cp in checkpoints}
        self._records: list[CheckpointRecord] = []

    def checkpoint(self, code: CheckpointCode) -> CapabilityCheckpoint:
        try:
            return self._cps[code]
        except KeyError as exc:
            raise CheckpointRegistryError(f"unknown checkpoint: {code!r}") from exc

    def all_checkpoints(self) -> list[CapabilityCheckpoint]:
        return [self._cps[c] for c in sorted(self._cps, key=lambda c: c.value)]

    def status_of(self, code: CheckpointCode) -> CheckpointStatus:
        hits = [r for r in self._records if r.checkpoint_code == code]
        return hits[-1].status if hits else CheckpointStatus.PENDING

    def latest_record(self, code: CheckpointCode) -> CheckpointRecord | None:
        hits = [r for r in self._records if r.checkpoint_code == code]
        return hits[-1] if hits else None

    def record(self, rec: CheckpointRecord) -> CheckpointRecord:
        """Append *rec* (flag-gated). Raises if disabled or code unknown."""
        if not capability_checkpoints_enabled():
            raise CheckpointRegistryError(
                "capability-checkpoint recording is disabled; "
                "set ARAGORA_CAPABILITY_CHECKPOINTS_ENABLED=1 to enable"
            )
        if rec.checkpoint_code not in self._cps:
            raise CheckpointRegistryError(f"unknown checkpoint code: {rec.checkpoint_code!r}")
        self._records.append(rec)
        return rec


_DEFAULTS = [
    (
        "CP-1",
        "Sustained substrate",
        4,
        None,
        "3 consecutive green BC-12 soaks (1 idle + 2 productive); "
        "no LaunchAgent respawn-failure incidents requiring human kickstart",
        "Pause AGT-* planning work; debug substrate self-healing",
    ),
    (
        "CP-2",
        "Live crux activation",
        4,
        "CP-1",
        "CruxDetector emits ranked CruxSet on >=20 real debates per week; "
        "at least one CruxSet linked to a follow-up issue or claim",
        "Reduce AGT scope to crux-only and re-evaluate",
    ),
    (
        "CP-3",
        "External truth signal",
        4,
        "CP-2",
        "Manifold + Metaculus integration produces >=100 resolved predictions "
        "per agent per week; calibration curve is stable",
        "Defer reputation wiring; debug prediction pipeline",
    ),
    (
        "CP-4",
        "Reputation flow live",
        4,
        "CP-3",
        "At least one agent has reputation delta drive a real "
        "dispatch-eligibility change in production",
        "Reduce to read-only reputation surface; revisit policy",
    ),
    (
        "CP-5",
        "Productivity-positive",
        4,
        "CP-4",
        "VIAH trends positive over rolling 4-week window without operator rescue spike",
        "Pause new boosters; consolidate existing layers",
    ),
]


def build_default_registry() -> CheckpointRegistry:
    """Return a registry with all 5 checkpoints (all start PENDING, no records)."""
    return CheckpointRegistry(
        [
            CapabilityCheckpoint(
                code=CheckpointCode(code),
                title=title,
                window_weeks=weeks,
                depends_on=CheckpointCode(dep) if dep else None,
                pass_condition=pass_cond,
                action_if_not_met=action,
            )
            for code, title, weeks, dep, pass_cond, action in _DEFAULTS
        ]
    )


__all__ = [
    "CapabilityCheckpoint",
    "CheckpointCode",
    "CheckpointRecord",
    "CheckpointRegistry",
    "CheckpointRegistryError",
    "CheckpointStatus",
    "build_default_registry",
    "capability_checkpoints_enabled",
]
