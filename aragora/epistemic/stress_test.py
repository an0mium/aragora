"""Adversarial world-state stress-test (DIC-25 / #6219).

Operator-curated catalog of plausible-future perturbations (CVE drops,
API rate-limit shifts, corpus revisions, dependency drops) that probe
proof-carrying code units *offline* and report fragility deltas before
reality invalidates them.

Flag gate
---------
``ARAGORA_STRESS_TEST_ENABLED`` must be truthy (``1``/``true``/``yes``/``on``)
for ``run_stress_test()`` to proceed.  Constructing ``StressPerturbation``
and ``FragilityReport`` objects is always safe; only the aggregate runner
requires the flag so report-only consumers can freely build objects for
inspection without enabling the gate.

Live queue effect
-----------------
None.  All results are report-only.  No issue creation, no queue mutation,
no dispatch changes.  Activation gate: DIC-20/21/22 production-green plus
the same proof-first Foreman gate that governs the entire DIC-23..28
dialectical-runtime-synthesis layer.

See
---
- ``docs/plans/EPISTEMIC_CI_AND_CRUX_ENGINE.md`` §DIC-25
- Issue: https://github.com/synaptent/aragora/issues/6219
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
    """Return True when the stress-test subsystem is enabled.

    Priority: ``override`` kwarg > ``ARAGORA_STRESS_TEST_ENABLED`` env var.
    Default: False.
    """
    if override is not None:
        return override
    raw = os.environ.get("ARAGORA_STRESS_TEST_ENABLED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class StressPerturbation:
    """A single plausible-future world-state perturbation.

    ``simulated_impact`` is the fraction of integrity lost (0.0-1.0) when
    this perturbation strikes an in-scope proof unit.  An empty
    ``affected_proof_unit_ids`` list means *all* units are in scope.
    """

    perturbation_id: str
    kind: PerturbationKind
    description: str
    simulated_impact: float = 0.0
    affected_claim_ids: list[str] = field(default_factory=list)
    affected_proof_unit_ids: list[str] = field(default_factory=list)
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation_id": self.perturbation_id,
            "kind": self.kind,
            "description": self.description,
            "simulated_impact": self.simulated_impact,
            "affected_claim_ids": self.affected_claim_ids,
            "affected_proof_unit_ids": self.affected_proof_unit_ids,
            "source": self.source,
        }


@dataclass
class FragilityReport:
    """Fragility assessment for one proof unit under one perturbation."""

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
    """Aggregate result of a stress-test run across all perturbations × units."""

    perturbations_tested: int
    proof_units_probed: int
    reports: list[FragilityReport] = field(default_factory=list)
    most_fragile_unit_id: str = ""
    max_fragility_delta: float = 0.0

    @property
    def high_fragility_units(self) -> list[FragilityReport]:
        """Reports where fragility_delta > 0.3 — warrant operator attention."""
        return [r for r in self.reports if r.fragility_delta > 0.3]

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbations_tested": self.perturbations_tested,
            "proof_units_probed": self.proof_units_probed,
            "reports": [r.to_dict() for r in self.reports],
            "most_fragile_unit_id": self.most_fragile_unit_id,
            "max_fragility_delta": self.max_fragility_delta,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ACTION_THRESHOLDS: tuple[tuple[float, float, str], ...] = (
    # (max_stressed_integrity, min_fragility_delta, action)
    # Ordered from most-severe to least; first match wins.
    (0.3, 0.0, "fail_closed"),
    (1.0, 0.4, "repair_required"),
    (1.0, 0.2, "monitor"),
)


def _recommended_action(fragility_delta: float, stressed_integrity: float) -> str:
    for max_stressed, min_delta, action in _ACTION_THRESHOLDS:
        if stressed_integrity <= max_stressed and fragility_delta >= min_delta:
            return action
    return "pass"


def _probe_unit(
    proof_unit_id: str,
    baseline_integrity: float,
    perturbation: StressPerturbation,
) -> FragilityReport:
    in_scope = (
        not perturbation.affected_proof_unit_ids
        or proof_unit_id in perturbation.affected_proof_unit_ids
    )
    if in_scope:
        stressed = round(max(0.0, baseline_integrity - perturbation.simulated_impact), 4)
        delta = round(baseline_integrity - stressed, 4)
        reason = f"{perturbation.kind}: {perturbation.description[:120]}"
    else:
        stressed = round(baseline_integrity, 4)
        delta = 0.0
        reason = "not in perturbation scope"

    return FragilityReport(
        proof_unit_id=proof_unit_id,
        perturbation_id=perturbation.perturbation_id,
        baseline_integrity=round(baseline_integrity, 4),
        stressed_integrity=stressed,
        fragility_delta=delta,
        reason=reason,
        recommended_action=_recommended_action(delta, stressed),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stress_test(
    perturbations: list[StressPerturbation],
    proof_unit_integrities: dict[str, float],
    *,
    enabled: bool | None = None,
) -> StressTestResult:
    """Run offline stress-tests for all perturbations × proof units.

    Args:
        perturbations: Operator-curated catalog of plausible-future
            perturbations.
        proof_unit_integrities: Mapping of ``proof_unit_id`` →
            current integrity score (0.0–1.0) from the DIC-20 decay
            monitor (``evaluate_unit``).
        enabled: Explicit override for the flag gate.  Pass ``True``
            in tests to bypass the env-var check.

    Returns:
        :class:`StressTestResult` with per-unit fragility reports.
        Always report-only; no queue mutation.

    Raises:
        RuntimeError: When the flag gate is off and ``enabled`` is not
            ``True``.
    """
    if not stress_test_enabled(override=enabled):
        raise RuntimeError(
            "Stress-test subsystem is disabled. "
            "Set ARAGORA_STRESS_TEST_ENABLED=1 to enable, "
            "or pass enabled=True in tests."
        )

    reports: list[FragilityReport] = []
    for perturbation in perturbations:
        for unit_id, baseline in proof_unit_integrities.items():
            reports.append(_probe_unit(unit_id, baseline, perturbation))

    if reports:
        max_delta = max(r.fragility_delta for r in reports)
        most_fragile = next(
            (r.proof_unit_id for r in reports if r.fragility_delta == max_delta),
            "",
        )
    else:
        max_delta = 0.0
        most_fragile = ""

    return StressTestResult(
        perturbations_tested=len(perturbations),
        proof_units_probed=len(proof_unit_integrities),
        reports=reports,
        most_fragile_unit_id=most_fragile,
        max_fragility_delta=round(max_delta, 4),
    )
