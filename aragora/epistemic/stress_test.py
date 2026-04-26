"""Adversarial world-state stress-test (DIC-25 / #6219).

Operator-curated perturbations probe proof-carrying code units *offline*
and report fragility deltas before reality invalidates them.
Flag: ``ARAGORA_STRESS_TEST_ENABLED`` (default off).  No queue mutation.
Issue: https://github.com/synaptent/aragora/issues/6219
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

PerturbationKind = Literal[
    "cve_drop",
    "api_rate_limit_shift",
    "corpus_revision",
    "dependency_drop",
    "claim_evidence_stale",
    "receipt_missing",
    "verifier_error",
]


def stress_test_enabled(*, override: bool | None = None) -> bool:
    """Return True when the gate is open (override kwarg > env var)."""
    if override is not None:
        return override
    return os.environ.get("ARAGORA_STRESS_TEST_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class StressPerturbation:
    """A plausible-future world-state perturbation.

    Empty ``affected_proof_unit_ids`` → all units in scope.
    ``simulated_impact`` is the integrity fraction lost (0.0–1.0).
    """

    perturbation_id: str
    kind: PerturbationKind
    description: str
    simulated_impact: float = 0.0
    affected_claim_ids: list[str] = field(default_factory=list)
    affected_proof_unit_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "kind": self.kind,
            "description": self.description,
            "simulated_impact": self.simulated_impact,
            "affected_claim_ids": self.affected_claim_ids,
            "affected_proof_unit_ids": self.affected_proof_unit_ids,
        }


@dataclass
class FragilityReport:
    """Fragility of one proof unit under one perturbation."""

    proof_unit_id: str
    perturbation_id: str
    baseline_integrity: float
    stressed_integrity: float
    fragility_delta: float
    reason: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_unit_id": self.proof_unit_id,
            "perturbation_id": self.perturbation_id,
            "baseline_integrity": self.baseline_integrity,
            "stressed_integrity": self.stressed_integrity,
            "fragility_delta": self.fragility_delta,
            "reason": self.reason,
            "recommended_action": self.recommended_action,
        }


@dataclass
class StressTestResult:
    """Aggregate result across all perturbations × proof units."""

    perturbations_tested: int
    proof_units_probed: int
    reports: list[FragilityReport] = field(default_factory=list)
    most_fragile_unit_id: str = ""
    max_fragility_delta: float = 0.0

    @property
    def high_fragility_units(self) -> list[FragilityReport]:
        """Reports with fragility_delta > 0.3 — warrant operator attention."""
        return [r for r in self.reports if r.fragility_delta > 0.3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbations_tested": self.perturbations_tested,
            "proof_units_probed": self.proof_units_probed,
            "reports": [r.to_dict() for r in self.reports],
            "most_fragile_unit_id": self.most_fragile_unit_id,
            "max_fragility_delta": self.max_fragility_delta,
        }


def _recommended_action(delta: float, stressed: float) -> str:
    if stressed < 0.3:
        return "fail_closed"
    if delta > 0.4:
        return "repair_required"
    if delta > 0.2:
        return "monitor"
    return "pass"


def _probe_unit(uid: str, baseline: float, p: StressPerturbation) -> FragilityReport:
    in_scope = not p.affected_proof_unit_ids or uid in p.affected_proof_unit_ids
    if in_scope:
        stressed = round(max(0.0, baseline - p.simulated_impact), 4)
        delta = round(baseline - stressed, 4)
        reason = f"{p.kind}: {p.description[:120]}"
    else:
        stressed, delta, reason = round(baseline, 4), 0.0, "not in perturbation scope"
    return FragilityReport(
        uid,
        p.perturbation_id,
        round(baseline, 4),
        stressed,
        delta,
        reason,
        _recommended_action(delta, stressed),
    )


def run_stress_test(
    perturbations: list[StressPerturbation],
    proof_unit_integrities: dict[str, float],
    *,
    enabled: bool | None = None,
) -> StressTestResult:
    """Run offline stress-tests for all perturbations × proof units.

    Args:
        perturbations: Operator-curated perturbation catalog.
        proof_unit_integrities: ``proof_unit_id`` → integrity score (0–1)
            from the DIC-20 decay monitor.
        enabled: Flag override; pass ``True`` in tests.

    Returns: Report-only :class:`StressTestResult`; no queue mutation.
    Raises: ``RuntimeError`` when gate is off and ``enabled`` is not ``True``.
    """
    if not stress_test_enabled(override=enabled):
        raise RuntimeError("Set ARAGORA_STRESS_TEST_ENABLED=1 to enable the stress-test subsystem.")
    reports = [
        _probe_unit(uid, baseline, p)
        for p in perturbations
        for uid, baseline in proof_unit_integrities.items()
    ]
    if reports:
        max_delta = max(r.fragility_delta for r in reports)
        most_fragile = next(r.proof_unit_id for r in reports if r.fragility_delta == max_delta)
    else:
        max_delta, most_fragile = 0.0, ""
    return StressTestResult(
        len(perturbations), len(proof_unit_integrities), reports, most_fragile, round(max_delta, 4)
    )
