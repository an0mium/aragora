from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


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
class DesignReviewRecord:
    manifest_id: str
    status: str
    rounds: list[dict[str, Any]] = field(default_factory=list)
    proposed_manifest: dict[str, Any] = field(default_factory=dict)
    critique_findings: list[str] = field(default_factory=list)
    revised_manifest: dict[str, Any] = field(default_factory=dict)
    unresolved_assumptions: list[str] = field(default_factory=list)
    recommendation: str = ""
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "status": self.status,
            "rounds": [dict(item) for item in self.rounds],
            "proposed_manifest": dict(self.proposed_manifest),
            "critique_findings": list(self.critique_findings),
            "revised_manifest": dict(self.revised_manifest),
            "unresolved_assumptions": list(self.unresolved_assumptions),
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DesignReviewRecord:
        data = data or {}
        return cls(
            manifest_id=str(data.get("manifest_id", "")).strip(),
            status=str(data.get("status", "")).strip(),
            rounds=[dict(item) for item in data.get("rounds", []) if isinstance(item, dict)],
            proposed_manifest=dict(data.get("proposed_manifest") or {}),
            critique_findings=_string_list(data.get("critique_findings")),
            revised_manifest=dict(data.get("revised_manifest") or {}),
            unresolved_assumptions=_string_list(data.get("unresolved_assumptions")),
            recommendation=str(data.get("recommendation", "")).strip(),
            created_at=_coerce_datetime(data.get("created_at")),
            updated_at=_coerce_datetime(data.get("updated_at")),
        )


async def run_design_review(
    *,
    manifest: Any,
    normalized_bundle: dict[str, Any],
    inspection: dict[str, Any],
    proposer_fn: Any | None = None,
    critic_fn: Any | None = None,
    synthesizer_fn: Any | None = None,
    max_rounds: int = 2,
) -> dict[str, Any]:
    rounds_limit = max(1, min(int(max_rounds or 2), 2))
    current_bundle = dict(normalized_bundle)
    rounds: list[dict[str, Any]] = []
    last_proposal: dict[str, Any] = {}
    last_findings: list[str] = []
    last_revised: dict[str, Any] = {}
    last_unresolved: list[str] = []
    final_recommendation = "approved"

    for round_number in range(1, rounds_limit + 1):
        proposal_payload = await _call_async(
            proposer_fn or _default_proposer,
            manifest=manifest,
            normalized_bundle=current_bundle,
            inspection=inspection,
            round_number=round_number,
            previous_findings=list(last_findings),
        )
        proposal = dict(proposal_payload.get("proposal") or current_bundle)

        critique_payload = await _call_async(
            critic_fn or _default_critic,
            manifest=manifest,
            normalized_bundle=current_bundle,
            inspection=inspection,
            proposal=proposal,
            round_number=round_number,
        )
        findings = _string_list(critique_payload.get("findings"))
        if findings and critique_payload.get("grounded") is False:
            raise ValueError(
                "design review critique findings must be grounded in manifest/ref/repo state"
            )

        synthesis_payload = await _call_async(
            synthesizer_fn or _default_synthesizer,
            manifest=manifest,
            normalized_bundle=current_bundle,
            inspection=inspection,
            proposal=proposal,
            critique=critique_payload,
            round_number=round_number,
            max_rounds=rounds_limit,
        )
        recommendation = (
            str(synthesis_payload.get("recommendation", "approved")).strip() or "approved"
        )
        revised_manifest = dict(synthesis_payload.get("revised_manifest") or proposal)
        unresolved_assumptions = _string_list(synthesis_payload.get("unresolved_assumptions"))
        rounds.append(
            {
                "round": round_number,
                "proposal": proposal,
                "findings": list(findings),
                "recommendation": recommendation,
                "revised_manifest": revised_manifest,
                "unresolved_assumptions": list(unresolved_assumptions),
            }
        )
        last_proposal = proposal
        last_findings = findings
        last_revised = revised_manifest
        last_unresolved = unresolved_assumptions
        if recommendation != "revise":
            final_recommendation = recommendation
            break
        current_bundle = dict(revised_manifest)
    else:
        final_recommendation = "awaiting_confirmation"

    if final_recommendation == "revise":
        final_recommendation = "awaiting_confirmation"

    record = DesignReviewRecord(
        manifest_id=str(getattr(manifest, "manifest_id", "")).strip(),
        status=final_recommendation,
        rounds=rounds,
        proposed_manifest=last_proposal,
        critique_findings=last_findings,
        revised_manifest=last_revised,
        unresolved_assumptions=last_unresolved,
        recommendation=final_recommendation,
    )
    return {
        "manifest_id": record.manifest_id,
        "recommendation": final_recommendation,
        "rounds_completed": len(rounds),
        "revised_manifest": dict(record.revised_manifest),
        "unresolved_assumptions": list(record.unresolved_assumptions),
        "record": record.to_dict(),
    }


def save_design_review(path: str | Path, record: DesignReviewRecord) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = record.to_dict()
    try:
        import yaml

        text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    except ImportError:
        text = json.dumps(payload, indent=2, sort_keys=False)
    target.write_text(text, encoding="utf-8")


def load_design_review(path: str | Path) -> DesignReviewRecord:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    try:
        import yaml

        payload = yaml.safe_load(text) or {}
    except ImportError:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Design review record must deserialize to an object.")
    return DesignReviewRecord.from_dict(payload)


async def _default_proposer(
    *,
    manifest: Any,
    normalized_bundle: dict[str, Any],
    inspection: dict[str, Any],
    round_number: int,
    previous_findings: list[str],
) -> dict[str, Any]:
    return {"proposal": dict(normalized_bundle)}


async def _default_critic(
    *,
    manifest: Any,
    normalized_bundle: dict[str, Any],
    inspection: dict[str, Any],
    proposal: dict[str, Any],
    round_number: int,
) -> dict[str, Any]:
    findings: list[str] = []
    if str(inspection.get("preflight_status", "")).strip() == "blocked":
        findings.extend(_string_list(inspection.get("preflight_blockers")))
    if not proposal.get("lanes"):
        findings.append("Normalized bundle has no lanes to execute.")
    return {"findings": findings, "grounded": True}


async def _default_synthesizer(
    *,
    manifest: Any,
    normalized_bundle: dict[str, Any],
    inspection: dict[str, Any],
    proposal: dict[str, Any],
    critique: dict[str, Any],
    round_number: int,
    max_rounds: int,
) -> dict[str, Any]:
    findings = _string_list(critique.get("findings"))
    if not findings:
        return {
            "recommendation": "approved",
            "revised_manifest": dict(proposal),
            "unresolved_assumptions": [],
        }
    recommendation = "needs_human" if round_number >= max_rounds else "awaiting_confirmation"
    return {
        "recommendation": recommendation,
        "revised_manifest": dict(proposal),
        "unresolved_assumptions": findings,
    }


async def _call_async(fn: Any, /, **kwargs: Any) -> dict[str, Any]:
    result = fn(**kwargs)
    if inspect.isawaitable(result):
        result = await result
    return dict(result or {})
