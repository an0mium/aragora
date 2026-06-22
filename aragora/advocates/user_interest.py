"""Local user-interest advocate interface and deterministic baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


AdvocateDecision = Literal["accept", "challenge", "ask_user", "block"]


@dataclass(frozen=True)
class AdvocateInput:
    """Input contract for a user-interest advocate."""

    task_type: str
    artifact_summary: str
    proposed_action: str
    context_features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdvocateOutput:
    """Output contract for a user-interest advocate."""

    decision: AdvocateDecision
    confidence: float
    rationale: str
    cited_features: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "cited_features": list(self.cited_features),
        }


class UserInterestAdvocate(Protocol):
    """Protocol implemented by local advocates and future adapters."""

    name: str

    def evaluate(self, request: AdvocateInput) -> AdvocateOutput:
        """Evaluate a proposed action against durable user interests."""


class RulesUserInterestAdvocate:
    """Deterministic baseline that encodes current repo safety posture."""

    name = "rules_user_interest_advocate"

    def evaluate(self, request: AdvocateInput) -> AdvocateOutput:
        features = request.context_features
        cited: list[str] = []

        tier = _as_int(features.get("tier"))
        if tier is not None and tier >= 3:
            cited.append("tier")
            return AdvocateOutput(
                decision="block",
                confidence=0.95,
                rationale="Tier 3/4 work requires explicit human-risk settlement.",
                cited_features=tuple(cited),
            )

        if _truthy(features.get("requires_human_risk_settlement")):
            cited.append("requires_human_risk_settlement")
            return AdvocateOutput(
                decision="block",
                confidence=0.95,
                rationale="Human-risk settlement is required before the proposed action.",
                cited_features=tuple(cited),
            )

        if _truthy(features.get("dirty")) or _truthy(features.get("conflicting")):
            cited.extend([key for key in ("dirty", "conflicting") if _truthy(features.get(key))])
            return AdvocateOutput(
                decision="challenge",
                confidence=0.85,
                rationale="Dirty or conflicting state should be repaired before action.",
                cited_features=tuple(cited),
            )

        failing_checks = _as_int(features.get("failing_checks")) or 0
        pending_checks = _as_int(features.get("pending_checks")) or 0
        if failing_checks > 0:
            cited.append("failing_checks")
            return AdvocateOutput(
                decision="block",
                confidence=0.9,
                rationale="Required or relevant checks are failing.",
                cited_features=tuple(cited),
            )
        if pending_checks > 0:
            cited.append("pending_checks")
            return AdvocateOutput(
                decision="ask_user",
                confidence=0.75,
                rationale="Wait for pending checks or explicit operator direction.",
                cited_features=tuple(cited),
            )

        changed_files = [str(item) for item in features.get("changed_files", []) or []]
        risky_paths = [
            path
            for path in changed_files
            if path.startswith(".github/")
            or "security" in path.lower()
            or "auth" in path.lower()
            or "secrets" in path.lower()
        ]
        if risky_paths:
            cited.append("changed_files")
            return AdvocateOutput(
                decision="challenge",
                confidence=0.82,
                rationale="Risk-sensitive paths need stronger review before action.",
                cited_features=tuple(cited),
            )

        if tier in (0, 1, 2) and str(request.proposed_action).lower() in {"merge", "accept"}:
            cited.append("tier")
            return AdvocateOutput(
                decision="accept",
                confidence=0.78,
                rationale="Low-risk clean work matches the current operator queue-drain posture.",
                cited_features=tuple(cited),
            )

        return AdvocateOutput(
            decision="ask_user",
            confidence=0.55,
            rationale="Insufficient signal to infer the operator's durable preference.",
            cited_features=tuple(cited),
        )


class LocalMockUserInterestAdvocate(RulesUserInterestAdvocate):
    """Mock local-model adapter used by AFT before any real fine-tuning exists."""

    name = "mock_local_user_interest_advocate"

    def evaluate(self, request: AdvocateInput) -> AdvocateOutput:
        output = super().evaluate(request)
        if output.decision != "ask_user":
            return output

        summary = f"{request.artifact_summary} {request.proposed_action}".lower()
        if any(term in summary for term in ("dependabot", "workflow", "tier 4")):
            return AdvocateOutput(
                decision="challenge",
                confidence=0.68,
                rationale="Mock local advocate detected a historical operator caution pattern.",
                cited_features=("artifact_summary",),
            )
        return output


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
