from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_VERSION = "pr_review_protocol.v1"
PROTOCOL_STATUS = "metadata_heuristic"

RECOMMEND_APPROVE = "approve_candidate"
RECOMMEND_ATTENTION = "needs_human_attention"
RECOMMEND_REPAIR = "repair_first"

REVIEW_ROLES: tuple[str, ...] = (
    "logic_reviewer",
    "security_reviewer",
    "maintainability_reviewer",
    "skeptic",
    "synthesizer",
)

_SLOT_CATALOG: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    ("logic", "logic_reviewer", "core", "claude", ("claude", "anthropic-api")),
    ("security", "security_reviewer", "core", "gpt", ("codex", "openai-api", "openai")),
    (
        "maintainability",
        "maintainability_reviewer",
        "heterodox",
        "gemini",
        ("gemini-cli", "gemini"),
    ),
    ("skeptic", "skeptic", "heterodox", "grok", ("grok-cli", "grok")),
    ("regulatory", "skeptic", "regulatory", "mistral", ("mistral-api", "mistral")),
)

_CLI_TO_BINARY = {
    "claude": "claude",
    "codex": "codex",
    "openai": "openai",
    "gemini-cli": "gemini",
    "grok-cli": "grok",
}

_ENV_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "anthropic-api": ("ANTHROPIC_API_KEY",),
    "openai-api": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "mistral-api": ("MISTRAL_API_KEY",),
    "mistral": ("OPENROUTER_API_KEY",),
}


@dataclass(frozen=True, slots=True)
class ProviderSlotDefinition:
    slot_id: str
    review_role: str
    lens: str
    family: str
    candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProviderSlotResolution:
    slot_id: str
    review_role: str
    lens: str
    family: str
    selected_provider: str | None
    status: str
    detail: str
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PRReviewBinding:
    repo: str
    pr_number: int
    base_sha: str
    head_sha: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PRReviewFinding:
    finding_id: str
    category: str
    severity: str
    summary: str
    evidence: list[str] = field(default_factory=list)
    source: str = PROTOCOL_STATUS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PRReviewProtocolPacket:
    protocol_version: str
    status: str
    binding: PRReviewBinding
    review_roles: list[str]
    provider_slots: list[ProviderSlotResolution]
    recommendation_class: str
    recommendation_reason: str
    confidence: float
    confidence_basis: str
    dissent_summary: str
    dissenting_views: list[dict[str, Any]]
    validation_summary: dict[str, Any]
    top_findings: list[PRReviewFinding]
    cost_estimate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["confidence"] = round(float(self.confidence), 2)
        return payload


@dataclass(frozen=True, slots=True)
class PRReviewProtocol:
    protocol_version: str = PROTOCOL_VERSION
    review_roles: tuple[str, ...] = REVIEW_ROLES
    provider_slots: tuple[ProviderSlotDefinition, ...] = field(default_factory=tuple)

    @classmethod
    def default(cls) -> PRReviewProtocol:
        return cls(
            provider_slots=tuple(
                ProviderSlotDefinition(
                    slot_id=slot_id,
                    review_role=review_role,
                    lens=lens,
                    family=family,
                    candidates=candidates,
                )
                for slot_id, review_role, lens, family, candidates in _SLOT_CATALOG
            )
        )

    def resolve_provider_slots(self) -> list[ProviderSlotResolution]:
        return [self._resolve_slot(slot) for slot in self.provider_slots]

    def build_packet(
        self,
        *,
        repo: str,
        pr_number: int,
        title: str,
        base_sha: str,
        head_sha: str,
        mergeable: str,
        review_decision: str,
        checks_summary: str,
        has_failures: bool,
        has_pending: bool,
        additions: int,
        deletions: int,
        changed_files: int,
        labels: list[str],
        high_risk_paths: list[str],
        validation_commands: list[str],
        machine_recommendation: str,
        machine_recommendation_reason: str,
    ) -> PRReviewProtocolPacket:
        provider_slots = self.resolve_provider_slots()
        findings = self._build_findings(
            has_failures=has_failures,
            has_pending=has_pending,
            mergeable=mergeable,
            additions=additions,
            deletions=deletions,
            labels=labels,
            high_risk_paths=high_risk_paths,
            checks_summary=checks_summary,
            title=title,
        )
        confidence = self._confidence_for(
            machine_recommendation=machine_recommendation,
            findings=findings,
            has_pending=has_pending,
        )
        slot_count = max(len(provider_slots), 1)
        cost_estimate = {
            "currency": "USD",
            "low": round(slot_count * 0.6, 2),
            "high": round(slot_count * 1.0, 2),
            "basis": "bounded heterogeneous metadata-first protocol",
        }
        validation_summary = {
            "checks_summary": checks_summary,
            "has_failures": has_failures,
            "has_pending": has_pending,
            "mergeable": mergeable,
            "review_decision": review_decision,
            "validation_commands": list(validation_commands),
            "changed_files": changed_files,
            "diffstat": {
                "additions": additions,
                "deletions": deletions,
            },
        }
        if machine_recommendation == RECOMMEND_APPROVE:
            dissent_summary = "No heterogeneous dissent recorded yet; packet is metadata-derived."
        else:
            dissent_summary = (
                "No heterogeneous dissent recorded yet; recommendation is driven by metadata risk "
                "signals pending full protocol execution."
            )
        return PRReviewProtocolPacket(
            protocol_version=self.protocol_version,
            status=PROTOCOL_STATUS,
            binding=PRReviewBinding(
                repo=repo,
                pr_number=pr_number,
                base_sha=base_sha,
                head_sha=head_sha,
            ),
            review_roles=list(self.review_roles),
            provider_slots=provider_slots,
            recommendation_class=machine_recommendation,
            recommendation_reason=machine_recommendation_reason,
            confidence=confidence,
            confidence_basis=PROTOCOL_STATUS,
            dissent_summary=dissent_summary,
            dissenting_views=[],
            validation_summary=validation_summary,
            top_findings=findings[:5],
            cost_estimate=cost_estimate,
        )

    def _resolve_slot(self, slot: ProviderSlotDefinition) -> ProviderSlotResolution:
        for candidate in slot.candidates:
            ok, detail = _provider_available(candidate)
            if ok:
                return ProviderSlotResolution(
                    slot_id=slot.slot_id,
                    review_role=slot.review_role,
                    lens=slot.lens,
                    family=slot.family,
                    selected_provider=candidate,
                    status="available",
                    detail=detail,
                    candidates=list(slot.candidates),
                )
        return ProviderSlotResolution(
            slot_id=slot.slot_id,
            review_role=slot.review_role,
            lens=slot.lens,
            family=slot.family,
            selected_provider=None,
            status="unavailable",
            detail=f"No configured provider available for {slot.family}",
            candidates=list(slot.candidates),
        )

    def _build_findings(
        self,
        *,
        has_failures: bool,
        has_pending: bool,
        mergeable: str,
        additions: int,
        deletions: int,
        labels: list[str],
        high_risk_paths: list[str],
        checks_summary: str,
        title: str,
    ) -> list[PRReviewFinding]:
        findings: list[PRReviewFinding] = []
        if has_failures:
            findings.append(
                PRReviewFinding(
                    finding_id="validation-failing",
                    category="validation",
                    severity="high",
                    summary="CI is failing on the current PR head.",
                    evidence=[checks_summary],
                )
            )
        if mergeable == "CONFLICTING":
            findings.append(
                PRReviewFinding(
                    finding_id="merge-conflict",
                    category="mergeability",
                    severity="high",
                    summary="PR has merge conflicts against the base branch.",
                    evidence=[mergeable],
                )
            )
        if high_risk_paths:
            findings.append(
                PRReviewFinding(
                    finding_id="high-risk-paths",
                    category="risk_surface",
                    severity="medium" if not has_failures else "high",
                    summary="PR touches high-consequence paths that warrant direct human review.",
                    evidence=high_risk_paths[:5],
                )
            )
        if additions + deletions > 500:
            findings.append(
                PRReviewFinding(
                    finding_id="large-diff",
                    category="change_size",
                    severity="medium",
                    summary="Diff size exceeds the bounded fast-review threshold.",
                    evidence=[f"+{additions}/-{deletions}"],
                )
            )
        if has_pending:
            findings.append(
                PRReviewFinding(
                    finding_id="checks-pending",
                    category="validation",
                    severity="medium",
                    summary="Not all required validation has completed yet.",
                    evidence=[checks_summary],
                )
            )
        if labels:
            parked = [
                label for label in labels if label in {"stale", "do-not-merge", "wip", "blocked"}
            ]
            if parked:
                findings.append(
                    PRReviewFinding(
                        finding_id="parked-label",
                        category="workflow_state",
                        severity="medium",
                        summary="PR carries a parked label and should remain in a human-attention lane.",
                        evidence=parked,
                    )
                )
        if not findings:
            findings.append(
                PRReviewFinding(
                    finding_id="bounded-green",
                    category="summary",
                    severity="low",
                    summary="No blocking metadata signals detected for this PR.",
                    evidence=[title, checks_summary],
                )
            )
        return findings

    def _confidence_for(
        self,
        *,
        machine_recommendation: str,
        findings: list[PRReviewFinding],
        has_pending: bool,
    ) -> float:
        if machine_recommendation == RECOMMEND_REPAIR:
            base = 0.9
        elif machine_recommendation == RECOMMEND_APPROVE:
            base = 0.72
        else:
            base = 0.58
        medium_or_higher = sum(1 for finding in findings if finding.severity in {"medium", "high"})
        penalty = min(0.18, medium_or_higher * 0.04)
        if has_pending and machine_recommendation != RECOMMEND_REPAIR:
            penalty += 0.04
        return max(0.15, min(0.98, round(base - penalty, 2)))


def default_pr_review_protocol() -> PRReviewProtocol:
    return PRReviewProtocol.default()


def _provider_available(candidate: str) -> tuple[bool, str]:
    binary = _CLI_TO_BINARY.get(candidate)
    if binary:
        if shutil.which(binary):
            return True, f"{binary} CLI available"
        return False, f"{binary} CLI not found on PATH"

    env_names = _ENV_REQUIREMENTS.get(candidate)
    if env_names:
        for env_name in env_names:
            if os.environ.get(env_name):
                return True, f"{env_name} configured"
        return False, f"missing env ({', '.join(env_names)})"

    return False, f"unsupported provider candidate: {candidate}"


__all__ = [
    "PRReviewBinding",
    "PRReviewFinding",
    "PRReviewProtocol",
    "PRReviewProtocolPacket",
    "PROTOCOL_STATUS",
    "PROTOCOL_VERSION",
    "ProviderSlotDefinition",
    "ProviderSlotResolution",
    "RECOMMEND_APPROVE",
    "RECOMMEND_ATTENTION",
    "RECOMMEND_REPAIR",
    "REVIEW_ROLES",
    "default_pr_review_protocol",
]
