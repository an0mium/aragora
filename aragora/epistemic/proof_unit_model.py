"""Pure Proof-Carrying Code Unit schema (DIC-19 / #6030).

Links a code path to the assumptions, evidence, claim IDs, verifier
commands, and decay/fallback policies that justify it. Schema-only and
read-only: no filesystem access, runtime mutation, quarantine, or issue
creation.

Field names for ``claims``, ``verifiers``, and ``decision_receipts`` are
aligned with the DIC-13 claim manifest schema and the AGT-01 CruxSet
receipt model so downstream tooling can join across all three.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_DECAY_ACTIONS = frozenset({"report_only", "repair_required", "fail_closed"})
_FALLBACK_ACTIONS = frozenset({"fail_closed", "degrade", "report_only"})


@dataclass
class DecayPolicy:
    failed_claim: str = "report_only"
    stale_evidence: str = "report_only"
    unresolved_crux: str = "report_only"

    def validate(self) -> list[str]:
        return [
            f"decay_policy.{attr}: invalid action {val!r}"
            for attr, val in (
                ("failed_claim", self.failed_claim),
                ("stale_evidence", self.stale_evidence),
                ("unresolved_crux", self.unresolved_crux),
            )
            if val not in _DECAY_ACTIONS
        ]


@dataclass
class FallbackPolicy:
    default: str = "fail_closed"
    operator_message: str = ""

    def validate(self) -> list[str]:
        if self.default not in _FALLBACK_ACTIONS:
            return [f"fallback_policy.default: invalid action {self.default!r}"]
        return []


@dataclass
class ProofCarryingCodeUnit:
    """Code path annotated with the proof that justifies it.

    Compatible with DIC-13 claim manifests (``claims`` IDs) and
    AGT-01 CruxSet receipts (``linked_crux_ids``).
    """

    code_unit_id: str
    symbol: str
    source_path: str
    owner: str
    decision_receipts: list[str]
    claims: list[str]
    assumptions: list[str]
    verifiers: list[dict[str, str]]
    freshness_sla_hours: int
    decay_policy: DecayPolicy
    fallback_policy: FallbackPolicy
    linked_crux_ids: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.code_unit_id:
            errors.append("code_unit_id must not be empty")
        if not self.source_path:
            errors.append("source_path must not be empty")
        if self.freshness_sla_hours < 1:
            errors.append(f"freshness_sla_hours must be >= 1, got {self.freshness_sla_hours}")
        errors.extend(self.decay_policy.validate())
        errors.extend(self.fallback_policy.validate())
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_unit_id": self.code_unit_id,
            "symbol": self.symbol,
            "source_path": self.source_path,
            "owner": self.owner,
            "decision_receipts": self.decision_receipts,
            "claims": self.claims,
            "assumptions": self.assumptions,
            "verifiers": self.verifiers,
            "freshness_sla_hours": self.freshness_sla_hours,
            "decay_policy": {
                "failed_claim": self.decay_policy.failed_claim,
                "stale_evidence": self.decay_policy.stale_evidence,
                "unresolved_crux": self.decay_policy.unresolved_crux,
            },
            "fallback_policy": {
                "default": self.fallback_policy.default,
                "operator_message": self.fallback_policy.operator_message,
            },
            "linked_crux_ids": self.linked_crux_ids,
        }


def load_proof_unit(data: dict[str, Any]) -> ProofCarryingCodeUnit:
    """Deserialise a dict (e.g. parsed YAML) into a :class:`ProofCarryingCodeUnit`."""
    decay = data.get("decay_policy") or {}
    fallback = data.get("fallback_policy") or {}
    return ProofCarryingCodeUnit(
        code_unit_id=data.get("code_unit_id", ""),
        symbol=data.get("symbol", ""),
        source_path=data.get("source_path", ""),
        owner=data.get("owner", ""),
        decision_receipts=list(data.get("decision_receipts") or []),
        claims=list(data.get("claims") or []),
        assumptions=list(data.get("assumptions") or []),
        verifiers=list(data.get("verifiers") or []),
        freshness_sla_hours=int(data.get("freshness_sla_hours", 24)),
        decay_policy=DecayPolicy(
            failed_claim=decay.get("failed_claim", "report_only"),
            stale_evidence=decay.get("stale_evidence", "report_only"),
            unresolved_crux=decay.get("unresolved_crux", "report_only"),
        ),
        fallback_policy=FallbackPolicy(
            default=fallback.get("default", "fail_closed"),
            operator_message=fallback.get("operator_message", ""),
        ),
        linked_crux_ids=list(data.get("linked_crux_ids") or []),
    )
