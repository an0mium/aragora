"""Mission lineage and gate primitives for the swarm substrate.

These models are intentionally additive. They annotate existing execution
contracts without replacing ``SwarmSpec`` or ``BoundedWorkOrder`` as sources
of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class GateType(str, Enum):
    DRAFT_READY = "draft_ready"
    DISPATCH_READY = "dispatch_ready"
    MILESTONE_READY = "milestone_ready"
    PUBLISH_READY = "publish_ready"


class GateVerdict(str, Enum):
    PASS = "pass"
    BLOCKED = "blocked"
    NEEDS_HUMAN = "needs_human"


class TranscriptAllowance(str, Enum):
    NONE = "none"
    SUMMARY_ONLY = "summary_only"
    RAW_ALLOWED = "raw_allowed"


def _ordered_unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


@dataclass(slots=True)
class MissionEnvelope:
    mission_id: str = ""
    roadmap_refs: list[str] = field(default_factory=list)
    goal_summary: str = ""
    assertion_ids: list[str] = field(default_factory=list)
    evidence_expectations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": str(self.mission_id or "").strip(),
            "roadmap_refs": _ordered_unique_strings(self.roadmap_refs),
            "goal_summary": str(self.goal_summary or "").strip(),
            "assertion_ids": _ordered_unique_strings(self.assertion_ids),
            "evidence_expectations": _ordered_unique_strings(self.evidence_expectations),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> MissionEnvelope:
        data = dict(payload or {})
        return cls(
            mission_id=str(data.get("mission_id", "") or "").strip(),
            roadmap_refs=_ordered_unique_strings(list(data.get("roadmap_refs") or [])),
            goal_summary=str(data.get("goal_summary", "") or "").strip(),
            assertion_ids=_ordered_unique_strings(list(data.get("assertion_ids") or [])),
            evidence_expectations=_ordered_unique_strings(
                list(data.get("evidence_expectations") or [])
            ),
        )


@dataclass(slots=True)
class RepairPolicy:
    max_repair_rounds: int = 2
    max_validator_rounds: int = 3
    max_stage_wall_time_minutes: int = 90
    escalate_after_terminal_classes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_repair_rounds": max(0, int(self.max_repair_rounds)),
            "max_validator_rounds": max(0, int(self.max_validator_rounds)),
            "max_stage_wall_time_minutes": max(0, int(self.max_stage_wall_time_minutes)),
            "escalate_after_terminal_classes": _ordered_unique_strings(
                self.escalate_after_terminal_classes
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> RepairPolicy:
        data = dict(payload or {})
        return cls(
            max_repair_rounds=int(data.get("max_repair_rounds", 2) or 2),
            max_validator_rounds=int(data.get("max_validator_rounds", 3) or 3),
            max_stage_wall_time_minutes=int(data.get("max_stage_wall_time_minutes", 90) or 90),
            escalate_after_terminal_classes=_ordered_unique_strings(
                list(data.get("escalate_after_terminal_classes") or [])
            ),
        )


@dataclass(slots=True)
class MissionStage:
    stage_id: str = ""
    mission_id: str = ""
    title: str = ""
    assertion_ids: list[str] = field(default_factory=list)
    file_scope: list[str] = field(default_factory=list)
    validation_command: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    repair_policy: RepairPolicy = field(default_factory=RepairPolicy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": str(self.stage_id or "").strip(),
            "mission_id": str(self.mission_id or "").strip(),
            "title": str(self.title or "").strip(),
            "assertion_ids": _ordered_unique_strings(self.assertion_ids),
            "file_scope": _ordered_unique_strings(self.file_scope),
            "validation_command": str(self.validation_command or "").strip(),
            "acceptance_criteria": _ordered_unique_strings(self.acceptance_criteria),
            "repair_policy": self.repair_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> MissionStage:
        data = dict(payload or {})
        return cls(
            stage_id=str(data.get("stage_id", "") or "").strip(),
            mission_id=str(data.get("mission_id", "") or "").strip(),
            title=str(data.get("title", "") or "").strip(),
            assertion_ids=_ordered_unique_strings(list(data.get("assertion_ids") or [])),
            file_scope=_ordered_unique_strings(list(data.get("file_scope") or [])),
            validation_command=str(data.get("validation_command", "") or "").strip(),
            acceptance_criteria=_ordered_unique_strings(
                list(data.get("acceptance_criteria") or [])
            ),
            repair_policy=RepairPolicy.from_dict(data.get("repair_policy")),
        )


@dataclass(slots=True)
class MissionContextPolicy:
    role: str
    allowed_artifact_classes: list[str] = field(default_factory=list)
    max_source_count: int = 0
    max_chars: int = 0
    freshness_ttl_seconds: int = 0
    transcript_allowance: str = TranscriptAllowance.NONE.value
    required_sources: list[str] = field(default_factory=list)
    forbidden_sources: list[str] = field(default_factory=list)

    def is_resolvable(self) -> bool:
        return bool(
            str(self.role or "").strip()
            and self.allowed_artifact_classes
            and self.max_source_count > 0
            and self.max_chars > 0
            and self.freshness_ttl_seconds >= 0
            and str(self.transcript_allowance or "").strip()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": str(self.role or "").strip(),
            "allowed_artifact_classes": _ordered_unique_strings(self.allowed_artifact_classes),
            "max_source_count": max(0, int(self.max_source_count)),
            "max_chars": max(0, int(self.max_chars)),
            "freshness_ttl_seconds": max(0, int(self.freshness_ttl_seconds)),
            "transcript_allowance": str(self.transcript_allowance or "").strip()
            or TranscriptAllowance.NONE.value,
            "required_sources": _ordered_unique_strings(self.required_sources),
            "forbidden_sources": _ordered_unique_strings(self.forbidden_sources),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> MissionContextPolicy:
        data = dict(payload or {})
        role = str(data.get("role", "") or "").strip() or "worker"
        return cls(
            role=role,
            allowed_artifact_classes=_ordered_unique_strings(
                list(data.get("allowed_artifact_classes") or [])
            ),
            max_source_count=int(data.get("max_source_count", 0) or 0),
            max_chars=int(data.get("max_chars", 0) or 0),
            freshness_ttl_seconds=int(data.get("freshness_ttl_seconds", 0) or 0),
            transcript_allowance=str(data.get("transcript_allowance", "") or "").strip()
            or TranscriptAllowance.NONE.value,
            required_sources=_ordered_unique_strings(list(data.get("required_sources") or [])),
            forbidden_sources=_ordered_unique_strings(list(data.get("forbidden_sources") or [])),
        )


@dataclass(slots=True)
class GateEvaluation:
    gate_type: str
    verdict: str
    mission_id: str = ""
    stage_id: str = ""
    assertion_ids: list[str] = field(default_factory=list)
    failure_classes: list[str] = field(default_factory=list)
    repair_eligible: bool = False
    required_evidence: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_type": str(self.gate_type or "").strip(),
            "verdict": str(self.verdict or "").strip(),
            "mission_id": str(self.mission_id or "").strip(),
            "stage_id": str(self.stage_id or "").strip(),
            "assertion_ids": _ordered_unique_strings(self.assertion_ids),
            "failure_classes": _ordered_unique_strings(self.failure_classes),
            "repair_eligible": bool(self.repair_eligible),
            "required_evidence": _ordered_unique_strings(self.required_evidence),
            "notes": str(self.notes or "").strip(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> GateEvaluation:
        data = dict(payload or {})
        return cls(
            gate_type=str(data.get("gate_type", "") or "").strip(),
            verdict=str(data.get("verdict", "") or "").strip(),
            mission_id=str(data.get("mission_id", "") or "").strip(),
            stage_id=str(data.get("stage_id", "") or "").strip(),
            assertion_ids=_ordered_unique_strings(list(data.get("assertion_ids") or [])),
            failure_classes=_ordered_unique_strings(list(data.get("failure_classes") or [])),
            repair_eligible=bool(data.get("repair_eligible", False)),
            required_evidence=_ordered_unique_strings(list(data.get("required_evidence") or [])),
            notes=str(data.get("notes", "") or "").strip(),
        )


def default_context_policy(
    role: str,
    *,
    file_scope: list[str] | None = None,
    evidence_expectations: list[str] | None = None,
) -> MissionContextPolicy:
    normalized_role = str(role or "").strip().lower() or "worker"
    scope = _ordered_unique_strings(list(file_scope or []))
    evidence = _ordered_unique_strings(list(evidence_expectations or []))

    if normalized_role == "validator":
        return MissionContextPolicy(
            role="validator",
            allowed_artifact_classes=[
                "mission_envelope",
                "mission_stage",
                "validation_command",
                "acceptance_criteria",
                "receipt",
                "artifact_ref",
                "scope_report",
                "summary",
            ],
            max_source_count=max(4, min(8, len(scope) + max(1, len(evidence)))),
            max_chars=16000,
            freshness_ttl_seconds=900,
            transcript_allowance=TranscriptAllowance.SUMMARY_ONLY.value,
            required_sources=evidence or ["validation_command"],
            forbidden_sources=["raw_worker_transcript"],
        )

    return MissionContextPolicy(
        role="worker",
        allowed_artifact_classes=[
            "mission_envelope",
            "mission_stage",
            "swarm_spec",
            "file_scope",
            "validation_command",
            "acceptance_criteria",
            "constraint",
            "knowledge_snippet",
            "evidence_expectation",
        ],
        max_source_count=max(4, min(12, len(scope) + 4)),
        max_chars=24000,
        freshness_ttl_seconds=3600,
        transcript_allowance=TranscriptAllowance.NONE.value,
        required_sources=scope[:6],
        forbidden_sources=["raw_worker_transcript", "validator_private_notes"],
    )


def normalize_context_policies(
    payload: Mapping[str, Any] | None,
    *,
    file_scope: list[str] | None = None,
    evidence_expectations: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    data = dict(payload or {})
    result: dict[str, dict[str, Any]] = {}
    for role in ("worker", "validator"):
        value = data.get(role)
        if isinstance(value, Mapping):
            policy = MissionContextPolicy.from_dict(value)
        else:
            policy = default_context_policy(
                role,
                file_scope=list(file_scope or []),
                evidence_expectations=list(evidence_expectations or []),
            )
        result[role] = policy.to_dict()
    return result


def mission_lineage_payload(
    *,
    mission_id: str = "",
    stage_id: str = "",
    assertion_ids: list[str] | None = None,
    roadmap_refs: list[str] | None = None,
    evidence_expectations: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "mission_id": str(mission_id or "").strip(),
        "stage_id": str(stage_id or "").strip(),
        "assertion_ids": _ordered_unique_strings(list(assertion_ids or [])),
        "roadmap_refs": _ordered_unique_strings(list(roadmap_refs or [])),
        "evidence_expectations": _ordered_unique_strings(list(evidence_expectations or [])),
    }


__all__ = [
    "GateEvaluation",
    "GateType",
    "GateVerdict",
    "MissionContextPolicy",
    "MissionEnvelope",
    "MissionStage",
    "RepairPolicy",
    "TranscriptAllowance",
    "default_context_policy",
    "mission_lineage_payload",
    "normalize_context_policies",
]
