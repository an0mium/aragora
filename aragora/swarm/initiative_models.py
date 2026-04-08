from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from aragora.pipeline.decision_plan.core import PlanStatus


DEFAULT_PLAN_STATUS = PlanStatus.CREATED.value


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InitiativeSlice:
    slice_id: str
    title: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    file_scope: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"
    status: str = DEFAULT_PLAN_STATUS
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "title": self.title,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "file_scope": list(self.file_scope),
            "acceptance_criteria": list(self.acceptance_criteria),
            "validations": list(self.validations),
            "estimated_complexity": self.estimated_complexity,
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InitiativeSlice:
        return cls(
            slice_id=str(payload.get("slice_id", "") or "").strip(),
            title=str(payload.get("title", "") or "").strip(),
            description=str(payload.get("description", "") or "").strip(),
            dependencies=[
                str(item).strip() for item in payload.get("dependencies", []) if str(item).strip()
            ],
            file_scope=[
                str(item).strip() for item in payload.get("file_scope", []) if str(item).strip()
            ],
            acceptance_criteria=[
                str(item).strip()
                for item in payload.get("acceptance_criteria", [])
                if str(item).strip()
            ],
            validations=[
                str(item).strip() for item in payload.get("validations", []) if str(item).strip()
            ],
            estimated_complexity=str(
                payload.get("estimated_complexity", "medium") or "medium"
            ).strip()
            or "medium",
            status=str(payload.get("status", DEFAULT_PLAN_STATUS) or DEFAULT_PLAN_STATUS).strip()
            or DEFAULT_PLAN_STATUS,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class InitiativeCheckpoint:
    checkpoint_id: str
    title: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    status: str = DEFAULT_PLAN_STATUS
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "title": self.title,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "validations": list(self.validations),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InitiativeCheckpoint:
        return cls(
            checkpoint_id=str(payload.get("checkpoint_id", "") or "").strip(),
            title=str(payload.get("title", "") or "").strip(),
            description=str(payload.get("description", "") or "").strip(),
            dependencies=[
                str(item).strip() for item in payload.get("dependencies", []) if str(item).strip()
            ],
            validations=[
                str(item).strip() for item in payload.get("validations", []) if str(item).strip()
            ],
            status=str(payload.get("status", DEFAULT_PLAN_STATUS) or DEFAULT_PLAN_STATUS).strip()
            or DEFAULT_PLAN_STATUS,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class InitiativeMilestone:
    milestone_id: str
    title: str
    description: str = ""
    slice_ids: list[str] = field(default_factory=list)
    checkpoint_ids: list[str] = field(default_factory=list)
    status: str = DEFAULT_PLAN_STATUS
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "milestone_id": self.milestone_id,
            "title": self.title,
            "description": self.description,
            "slice_ids": list(self.slice_ids),
            "checkpoint_ids": list(self.checkpoint_ids),
            "status": self.status,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InitiativeMilestone:
        return cls(
            milestone_id=str(payload.get("milestone_id", "") or "").strip(),
            title=str(payload.get("title", "") or "").strip(),
            description=str(payload.get("description", "") or "").strip(),
            slice_ids=[
                str(item).strip() for item in payload.get("slice_ids", []) if str(item).strip()
            ],
            checkpoint_ids=[
                str(item).strip() for item in payload.get("checkpoint_ids", []) if str(item).strip()
            ],
            status=str(payload.get("status", DEFAULT_PLAN_STATUS) or DEFAULT_PLAN_STATUS).strip()
            or DEFAULT_PLAN_STATUS,
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass
class InitiativeRecord:
    initiative_id: str
    title: str
    goal: str
    rationale: str
    slices: list[InitiativeSlice] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    feature_flag_name: str | None = None
    milestones: list[InitiativeMilestone] = field(default_factory=list)
    checkpoints: list[InitiativeCheckpoint] = field(default_factory=list)
    status: str = DEFAULT_PLAN_STATUS
    planner_rationale: str = ""
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utcnow_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "initiative_id": self.initiative_id,
            "title": self.title,
            "goal": self.goal,
            "rationale": self.rationale,
            "slices": [item.to_dict() for item in self.slices],
            "dependencies": list(self.dependencies),
            "validations": list(self.validations),
            "feature_flag_name": self.feature_flag_name,
            "milestones": [item.to_dict() for item in self.milestones],
            "checkpoints": [item.to_dict() for item in self.checkpoints],
            "status": self.status,
            "planner_rationale": self.planner_rationale,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InitiativeRecord:
        return cls(
            initiative_id=str(payload.get("initiative_id", "") or "").strip(),
            title=str(payload.get("title", "") or "").strip(),
            goal=str(payload.get("goal", "") or "").strip(),
            rationale=str(payload.get("rationale", "") or "").strip(),
            slices=[
                InitiativeSlice.from_dict(item)
                for item in payload.get("slices", [])
                if isinstance(item, dict)
            ],
            dependencies=[
                str(item).strip() for item in payload.get("dependencies", []) if str(item).strip()
            ],
            validations=[
                str(item).strip() for item in payload.get("validations", []) if str(item).strip()
            ],
            feature_flag_name=(str(payload.get("feature_flag_name", "") or "").strip() or None),
            milestones=[
                InitiativeMilestone.from_dict(item)
                for item in payload.get("milestones", [])
                if isinstance(item, dict)
            ],
            checkpoints=[
                InitiativeCheckpoint.from_dict(item)
                for item in payload.get("checkpoints", [])
                if isinstance(item, dict)
            ],
            status=str(payload.get("status", DEFAULT_PLAN_STATUS) or DEFAULT_PLAN_STATUS).strip()
            or DEFAULT_PLAN_STATUS,
            planner_rationale=str(payload.get("planner_rationale", "") or "").strip(),
            created_at=str(payload.get("created_at", "") or "").strip() or utcnow_iso(),
            updated_at=str(payload.get("updated_at", "") or "").strip() or utcnow_iso(),
            metadata=dict(payload.get("metadata") or {}),
        )
