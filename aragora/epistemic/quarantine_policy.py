"""Fail-closed quarantine policy for proof-carrying code units (DIC-21 / #6032).

Translates a :class:`~aragora.epistemic.decay_monitor.DecaySignal` into a
concrete :class:`QuarantineDecision`. Three invariants are non-negotiable:

1. Live routing is always blocked unless the unit ID is in a non-empty
   ``live_swap_allowlist``.
2. Integrity below ``fail_closed_threshold`` always produces ``"fail_closed"``.
3. Every non-``report_only`` decision carries a deterministic SHA-256
   provenance hash for downstream receipt generation.

Flag gate: ``ARAGORA_QUARANTINE_POLICY_ENABLED`` (default off).
Building and inspecting decisions is always safe; acting on them must be
gated. DIC-22 (Verified Replacement Pipeline) consumes
:class:`QuarantineDecision` to produce bounded repair specs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from aragora.epistemic.decay_monitor import DecaySignal

PolicyAction = Literal[
    "report_only", "degrade", "fallback", "quarantine", "repair_required", "fail_closed"
]

_RANK: dict[str, int] = {
    "report_only": 0,
    "degrade": 1,
    "fallback": 2,
    "quarantine": 3,
    "repair_required": 4,
    "fail_closed": 5,
}
_RANK_TO_ACTION: dict[int, PolicyAction] = {v: k for k, v in _RANK.items()}  # type: ignore[misc]


@dataclass(frozen=True)
class EscalationMap:
    report_only: PolicyAction = "report_only"
    repair_required: PolicyAction = "repair_required"
    fail_closed: PolicyAction = "fail_closed"

    def resolve(self, action: str) -> PolicyAction:
        return getattr(self, action, "fail_closed")  # type: ignore[return-value]


@dataclass(frozen=True)
class QuarantinePolicy:
    code_unit_class: str = "default"
    escalation_map: EscalationMap = field(default_factory=EscalationMap)
    fail_closed_threshold: float = 0.4
    live_swap_allowlist: frozenset[str] = field(default_factory=frozenset)


DEFAULT_POLICIES: dict[str, QuarantinePolicy] = {
    "live_dispatch": QuarantinePolicy(
        code_unit_class="live_dispatch",
        escalation_map=EscalationMap(report_only="report_only", repair_required="quarantine"),
        fail_closed_threshold=0.6,
    ),
    "report_surface": QuarantinePolicy(
        code_unit_class="report_surface",
        escalation_map=EscalationMap(report_only="report_only", repair_required="degrade"),
        fail_closed_threshold=0.3,
    ),
    "demo": QuarantinePolicy(
        code_unit_class="demo",
        escalation_map=EscalationMap(report_only="report_only", repair_required="fallback"),
        fail_closed_threshold=0.2,
    ),
    "pure_policy": QuarantinePolicy(code_unit_class="pure_policy"),
    "default": QuarantinePolicy(),
}


@dataclass
class QuarantineDecision:
    code_unit_id: str
    policy_action: PolicyAction
    rationale: str
    provenance_hash: str  # SHA-256; empty for report_only
    fail_closed: bool
    live_swap_blocked: bool
    integrity_score: float

    def to_dict(self) -> dict:
        return {
            "code_unit_id": self.code_unit_id,
            "policy_action": self.policy_action,
            "rationale": self.rationale,
            "provenance_hash": self.provenance_hash,
            "fail_closed": self.fail_closed,
            "live_swap_blocked": self.live_swap_blocked,
            "integrity_score": self.integrity_score,
        }


def apply_quarantine_policy(
    signal: "DecaySignal",
    policy: QuarantinePolicy | None = None,
    *,
    code_unit_class: str = "default",
    request_live_swap: bool = False,
) -> QuarantineDecision:
    """Apply policy to a decay signal. Pure function — no side effects."""
    if policy is None:
        policy = DEFAULT_POLICIES.get(code_unit_class, DEFAULT_POLICIES["default"])

    uid, score = signal.code_unit_id, signal.integrity_score
    is_fail_closed = score < policy.fail_closed_threshold

    if is_fail_closed:
        action: PolicyAction = "fail_closed"
        rationale = (
            f"Integrity {score:.3f} < threshold {policy.fail_closed_threshold:.3f} "
            f"(class={policy.code_unit_class!r})."
        )
    else:
        action = policy.escalation_map.resolve(signal.recommended_action)
        rationale = (
            f"Signal recommended {signal.recommended_action!r}; "
            f"escalated to {action!r} (class={policy.code_unit_class!r})."
        )

    live_swap_blocked = True
    if request_live_swap and policy.live_swap_allowlist and uid in policy.live_swap_allowlist:
        live_swap_blocked = False
        rationale += f" Live routing permitted (allowlist: {uid!r})."
    elif request_live_swap:
        if _RANK.get(action, 0) < _RANK["quarantine"]:
            action = "quarantine"
        rationale += f" Live routing blocked: {uid!r} not in allowlist."

    if action != "report_only":
        material = json.dumps(
            {
                "code_unit_id": uid,
                "integrity_score": score,
                "recommended_action": signal.recommended_action,
                "policy_action": action,
                "reason_kinds": sorted({r.kind for r in signal.reasons}),
            },
            sort_keys=True,
        )
        provenance_hash = hashlib.sha256(material.encode()).hexdigest()
    else:
        provenance_hash = ""

    return QuarantineDecision(
        code_unit_id=uid,
        policy_action=action,
        rationale=rationale,
        provenance_hash=provenance_hash,
        fail_closed=is_fail_closed,
        live_swap_blocked=live_swap_blocked,
        integrity_score=score,
    )


def quarantine_policy_enabled() -> bool:
    """Return True when ``ARAGORA_QUARANTINE_POLICY_ENABLED`` is set. Default: False."""
    return str(os.environ.get("ARAGORA_QUARANTINE_POLICY_ENABLED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
