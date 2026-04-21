"""Per-reviewer output contract for heterogeneous PR review execution.

Schema layer for the execution-path follow-on to #6306. This module defines
the structured payload each reviewer must emit before synthesis. It is still
behavior-light by design: no model calls, no storage, no orchestration. The
only logic here is contract validation and dict normalization so later slices
can reject malformed reviewer payloads deterministically.

Design source:
  docs/plans/2026-04-20-pr-review-execution-path.md

Contract boundary:
  - one ``ReviewerOutput`` is one reviewer's structured payload for one round
  - payloads are advisory inputs to synthesis, not settlement actions
  - evidence refs reuse the stable receipt-layer ``EvidenceRef`` contract
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from aragora.review.protocol import Recommendation
from aragora.review.receipt import EvidenceKind, EvidenceRef

REVIEWER_OUTPUT_SCHEMA_VERSION = "reviewer_output.v1"


class FindingCategory(str, Enum):
    """Allowed categories for reviewer findings.

    Values mirror the execution-path design doc's minimum reviewer payload.
    Keeping them as enums prevents silent drift across executor, synthesis,
    receipt, and UI layers.
    """

    LOGIC = "logic"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    SKEPTIC = "skeptic"
    VALIDATION = "validation"


class FindingSeverity(str, Enum):
    """Severity scale for reviewer findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _normalize_string_items(values: Sequence[Any] | None) -> tuple[str, ...]:
    items: list[str] = []
    seen: set[str] = set()
    for raw in list(values or []):
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
    return tuple(items)


def _parse_enum(enum_cls: type[Enum], value: Any, field_name: str) -> Enum:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"{field_name} is required")
    try:
        return enum_cls(raw)
    except ValueError as exc:
        allowed = ", ".join(sorted(member.value for member in enum_cls))  # type: ignore[attr-defined]
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _line_range_from_value(value: Any) -> tuple[int, int] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start = int(value[0])
        end = int(value[1])
        return (start, end)
    raise ValueError("line_range must be a 2-item list or tuple when provided")


def _evidence_ref_from_dict(data: Mapping[str, Any]) -> EvidenceRef:
    return EvidenceRef(
        kind=_parse_enum(EvidenceKind, data.get("kind"), "evidence_refs[].kind"),  # type: ignore[arg-type]
        path=str(data.get("path", "") or "").strip(),
        sha=str(data.get("sha", "") or "").strip(),
        line_range=_line_range_from_value(data.get("line_range")),
        quote=str(data.get("quote", "") or "").strip(),
    )


@dataclass(frozen=True, slots=True)
class ReviewerFinding:
    """One structured finding from a reviewer output."""

    category: FindingCategory
    severity: FindingSeverity
    claim: str
    evidence: tuple[str, ...]
    files: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        d["severity"] = self.severity.value
        d["evidence"] = list(self.evidence)
        d["files"] = list(self.files)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewerFinding":
        finding = cls(
            category=_parse_enum(FindingCategory, data.get("category"), "top_findings[].category"),  # type: ignore[arg-type]
            severity=_parse_enum(FindingSeverity, data.get("severity"), "top_findings[].severity"),  # type: ignore[arg-type]
            claim=str(data.get("claim", "") or "").strip(),
            evidence=_normalize_string_items(data.get("evidence")),
            files=_normalize_string_items(data.get("files")),
        )
        finding.validate()
        return finding

    def validate(self) -> None:
        if not str(self.claim or "").strip():
            raise ValueError("top_findings[].claim is required")
        if not self.evidence:
            raise ValueError("top_findings[].evidence must contain at least one item")
        if self.category != FindingCategory.VALIDATION and not self.files:
            raise ValueError(
                "top_findings[].files must contain at least one path for non-validation findings"
            )


@dataclass(frozen=True, slots=True)
class ReviewerOutput:
    """Structured output emitted by one reviewer for one round."""

    reviewer_id: str
    slot_id: str
    provider: str
    lens: str
    family: str
    recommendation_class: Recommendation
    confidence: float
    summary: str
    top_findings: tuple[ReviewerFinding, ...]
    evidence_refs: tuple[EvidenceRef, ...]
    risk_flags: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    round_index: int = 1
    latency_ms: int = 0
    cost_usd: float = 0.0
    schema_version: str = REVIEWER_OUTPUT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reviewer_id": self.reviewer_id,
            "slot_id": self.slot_id,
            "provider": self.provider,
            "lens": self.lens,
            "family": self.family,
            "recommendation_class": self.recommendation_class.value,
            "confidence": round(float(self.confidence), 4),
            "summary": self.summary,
            "top_findings": [finding.to_dict() for finding in self.top_findings],
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
            "risk_flags": list(self.risk_flags),
            "open_questions": list(self.open_questions),
            "round_index": self.round_index,
            "latency_ms": self.latency_ms,
            "cost_usd": round(float(self.cost_usd), 6),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewerOutput":
        output = cls(
            schema_version=str(
                data.get("schema_version", REVIEWER_OUTPUT_SCHEMA_VERSION)
                or REVIEWER_OUTPUT_SCHEMA_VERSION
            ).strip(),
            reviewer_id=str(data.get("reviewer_id", "") or "").strip(),
            slot_id=str(data.get("slot_id", "") or "").strip(),
            provider=str(data.get("provider", "") or "").strip(),
            lens=str(data.get("lens", "") or "").strip(),
            family=str(data.get("family", "") or "").strip(),
            recommendation_class=_parse_enum(
                Recommendation,
                data.get("recommendation_class"),
                "recommendation_class",
            ),  # type: ignore[arg-type]
            confidence=float(data.get("confidence", 0.0) or 0.0),
            summary=str(data.get("summary", "") or "").strip(),
            top_findings=tuple(
                ReviewerFinding.from_dict(item)
                for item in list(data.get("top_findings", []) or [])
                if isinstance(item, Mapping)
            ),
            evidence_refs=tuple(
                _evidence_ref_from_dict(item)
                for item in list(data.get("evidence_refs", []) or [])
                if isinstance(item, Mapping)
            ),
            risk_flags=_normalize_string_items(data.get("risk_flags")),
            open_questions=_normalize_string_items(data.get("open_questions")),
            round_index=int(data.get("round_index", 1) or 1),
            latency_ms=int(data.get("latency_ms", 0) or 0),
            cost_usd=float(data.get("cost_usd", 0.0) or 0.0),
        )
        output.validate()
        return output

    def validate(self) -> None:
        required = {
            "schema_version": self.schema_version,
            "reviewer_id": self.reviewer_id,
            "slot_id": self.slot_id,
            "provider": self.provider,
            "lens": self.lens,
            "family": self.family,
            "summary": self.summary,
        }
        missing = [key for key, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(f"Reviewer output missing required fields: {', '.join(missing)}")
        if self.schema_version != REVIEWER_OUTPUT_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {REVIEWER_OUTPUT_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.round_index < 1:
            raise ValueError("round_index must be >= 1")
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be >= 0")
        if self.cost_usd < 0:
            raise ValueError("cost_usd must be >= 0")
        if not self.top_findings:
            raise ValueError("top_findings must contain at least one item")
        if not self.evidence_refs:
            raise ValueError("evidence_refs must contain at least one item")
        for finding in self.top_findings:
            finding.validate()


def validate_reviewer_outputs(outputs: Sequence[ReviewerOutput]) -> None:
    """Validate a batch of reviewer outputs.

    The batch contract is intentionally minimal at this layer:
      - at least one output must be present
      - each output must satisfy ``ReviewerOutput.validate()``
      - one reviewer cannot emit two payloads for the same round
    """

    if not outputs:
        raise ValueError("at least one reviewer output is required")

    seen: set[tuple[str, int]] = set()
    for output in outputs:
        output.validate()
        identity = (output.reviewer_id, output.round_index)
        if identity in seen:
            raise ValueError("duplicate reviewer output for the same reviewer_id and round_index")
        seen.add(identity)


__all__ = [
    "FindingCategory",
    "FindingSeverity",
    "REVIEWER_OUTPUT_SCHEMA_VERSION",
    "ReviewerFinding",
    "ReviewerOutput",
    "validate_reviewer_outputs",
]
