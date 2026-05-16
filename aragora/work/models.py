"""Normalized work-board models for the read-only Agent Flywheel kernel."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "aragora.work.v1"


@dataclass(slots=True)
class WorkScore:
    """Flywheel-inspired score dimensions.

    All dimensions are normalized to ``[0, 1]`` where higher means "more
    actionable now" for the operator/broker loop. ``risk`` therefore means
    risk-controlled / safely settleable, not raw danger.
    """

    readiness: float = 0.0
    impact: float = 0.0
    risk: float = 0.0
    parallel_safety: float = 0.0
    staleness: float = 0.0
    owner_clarity: float = 0.0
    test_obligation: float = 0.0
    dependency_clarity: float = 0.0
    bead_quality: float = 0.0
    total: float = 0.0
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 4)
        return data


@dataclass(slots=True)
class WorkItem:
    """A single normalized work unit from PRs, outbox, beads, runs, or missions."""

    id: str
    source: str
    item_type: str
    title: str
    status: str = "unknown"
    scope: str = "current"
    url: str | None = None
    owner: str | None = None
    branch: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    dependencies: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    score: WorkScore | None = None

    def to_dict(self, *, include_score: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if self.score and include_score:
            data["score"] = self.score.to_dict()
        elif not include_score:
            data.pop("score", None)
        return data


@dataclass(slots=True)
class WorkGraph:
    """Graph view over normalized work items."""

    items: list[WorkItem] = field(default_factory=list)
    edges: list[dict[str, str]] = field(default_factory=list)
    source_health: list[dict[str, Any]] = field(default_factory=list)
    root_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "root_id": self.root_id,
            "items": [item.to_dict() for item in self.items],
            "edges": list(self.edges),
            "source_health": list(self.source_health),
        }


@dataclass(slots=True)
class WorkRecommendation:
    """Actionable read-only recommendation for a future broker loop."""

    rank: int
    item_id: str
    classification: str
    action: str
    priority: str
    rationale: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    score: WorkScore = field(default_factory=WorkScore)
    item: WorkItem | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "item_id": self.item_id,
            "classification": self.classification,
            "action": self.action,
            "priority": self.priority,
            "rationale": list(self.rationale),
            "blockers": list(self.blockers),
            "score": self.score.to_dict(),
            "item": self.item.to_dict(include_score=False) if self.item else None,
        }
