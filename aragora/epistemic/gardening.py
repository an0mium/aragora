"""Proactive crux gardening — scheduled re-examination pass (DIC-28 / #6222).

Re-examines resolved and outstanding cruxes for:
- Evidence staleness (via DIC-14 ClaimResult status fields)
- New contradictions (via DIC-26 CoherenceIssue)
- Fragility score shifts (via DIC-25 fragility_score deltas)

Default: **OFF**. Set ``ARAGORA_CRUX_GARDENING_ENABLED=1`` to enable.
Report-only by default; DIC-17 follow-up feed is an opt-in flag.
No queue mutation, no auto-debate, no auto-issue creation.

Issue: https://github.com/synaptent/aragora/issues/6222
Gate: same proof-first Foreman gate as DIC-23..28.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from aragora.epistemic.claim_verifier import ClaimResult, ClaimStatus
from aragora.epistemic.coherence import CoherenceIssue, IncoherenceKind
from aragora.epistemic.crux_receipt import CruxEntry, CruxReceipt

_TRUTHY = frozenset({"1", "true", "yes", "on"})

# A fragility shift larger than this threshold surfaces as a "fragility_shift" finding.
DEFAULT_FRAGILITY_SHIFT_THRESHOLD: float = 0.15

GardeningStatus = str  # "healthy" | "stale_evidence" | "new_contradiction" | "fragility_shift"


def crux_gardening_enabled(*, override: bool | None = None) -> bool:
    """Return True when the DIC-28 gardening pass is enabled.

    Reads ``ARAGORA_CRUX_GARDENING_ENABLED``; default False.
    Override kwarg takes precedence for tests.
    """
    if override is not None:
        return override
    return os.environ.get("ARAGORA_CRUX_GARDENING_ENABLED", "").strip().lower() in _TRUTHY


def enable_crux_gardening() -> None:
    """Enable the DIC-28 gardening pass for the current process."""
    os.environ["ARAGORA_CRUX_GARDENING_ENABLED"] = "1"


@dataclass(frozen=True)
class CruxGardeningResult:
    """Per-crux finding from one gardening pass.

    ``status`` is one of: ``healthy``, ``stale_evidence``,
    ``new_contradiction``, ``fragility_shift``.

    ``needs_followup`` is True only when the DIC-17 bridge is explicitly
    enabled (``ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED``) and the status is
    not ``healthy``.
    """

    crux_id: str
    status: GardeningStatus
    detail: str
    previous_fragility: float | None = None
    current_fragility: float | None = None
    coherence_issue_kinds: tuple[str, ...] = field(default_factory=tuple)
    needs_followup: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["coherence_issue_kinds"] = list(self.coherence_issue_kinds)
        return d


@dataclass(frozen=True)
class GardeningReport:
    """Deterministic report from a full proactive crux gardening pass.

    ``generated_at`` is an ISO-8601 UTC timestamp.
    ``summary`` counts outcomes across both resolved and outstanding sets.
    """

    generated_at: str
    resolved_results: list[CruxGardeningResult]
    outstanding_results: list[CruxGardeningResult]
    summary: dict[str, int]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "resolved_results": [r.to_dict() for r in self.resolved_results],
            "outstanding_results": [r.to_dict() for r in self.outstanding_results],
            "summary": dict(self.summary),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _followup_eligible() -> bool:
    raw = os.environ.get("ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED", "").strip().lower()
    return raw in _TRUTHY


def _coherence_kinds_for_crux(
    claim_ids: list[str],
    coherence_issues: list[CoherenceIssue],
) -> tuple[str, ...]:
    """Return IncoherenceKind values from coherence issues that touch this crux's claims."""
    crux_claim_set = frozenset(claim_ids)
    kinds: list[str] = []
    for issue in coherence_issues:
        affected = frozenset(issue.belief_ids)
        if affected & crux_claim_set:
            kinds.append(issue.kind.value)
    return tuple(dict.fromkeys(kinds))  # deduplicate, preserve order


def garden_resolved_crux(
    receipt: CruxReceipt,
    *,
    claim_results: dict[str, ClaimResult] | None = None,
    coherence_issues: list[CoherenceIssue] | None = None,
    followup_enabled: bool | None = None,
) -> list[CruxGardeningResult]:
    """Examine all cruxes in a resolved CruxReceipt for staleness or new contradictions.

    Returns one :class:`CruxGardeningResult` per :class:`CruxEntry` in the
    receipt.  Staleness is detected by finding a ``fail`` or ``stale``
    ClaimResult for any affected claim.  Contradictions are detected via
    DIC-26 coherence issues that reference the crux's affected claims.

    ``followup_enabled`` overrides the ``ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED``
    env-var read when provided; pass it explicitly from :func:`run_gardening_pass`
    so that domain logic doesn't read env directly on every call.
    """
    results: list[CruxGardeningResult] = []
    cr = claim_results or {}
    ci = coherence_issues or []
    _followup = followup_enabled if followup_enabled is not None else _followup_eligible()

    for entry in receipt.cruxes:
        stale_claims: list[str] = []
        for claim_id in entry.affected_claims:
            result = cr.get(claim_id)
            if result is not None and result.status in (ClaimStatus.STALE, ClaimStatus.FAIL):
                stale_claims.append(claim_id)

        coh_kinds = _coherence_kinds_for_crux(entry.affected_claims, ci)
        has_contradiction = IncoherenceKind.CONTRADICTION.value in coh_kinds

        if stale_claims:
            status: GardeningStatus = "stale_evidence"
            detail = f"stale or failed claims: {', '.join(stale_claims)}"
        elif has_contradiction:
            status = "new_contradiction"
            detail = f"coherence issues detected: {', '.join(coh_kinds)}"
        else:
            status = "healthy"
            detail = "evidence fresh; no new contradictions"

        needs_followup = _followup and status != "healthy"
        results.append(
            CruxGardeningResult(
                crux_id=entry.crux_id,
                status=status,
                detail=detail,
                coherence_issue_kinds=coh_kinds,
                needs_followup=needs_followup,
            )
        )
    return results


def garden_outstanding_crux(
    entry: CruxEntry,
    *,
    previous_fragility: float | None,
    current_fragility: float | None,
    fragility_shift_threshold: float = DEFAULT_FRAGILITY_SHIFT_THRESHOLD,
    followup_enabled: bool | None = None,
) -> CruxGardeningResult:
    """Check whether an outstanding crux has materially shifted in fragility.

    A crux surfaces as ``fragility_shift`` when the absolute delta between
    ``previous_fragility`` and ``current_fragility`` exceeds
    ``fragility_shift_threshold``.  When either value is None the crux
    cannot be compared and is marked ``healthy``.

    ``followup_enabled`` overrides the ``ARAGORA_EPISTEMIC_FOLLOWUP_ENABLED``
    env-var read when provided.
    """
    if previous_fragility is None or current_fragility is None:
        return CruxGardeningResult(
            crux_id=entry.crux_id,
            status="healthy",
            detail="fragility baseline unavailable; skipping comparison",
            previous_fragility=previous_fragility,
            current_fragility=current_fragility,
        )

    _followup = followup_enabled if followup_enabled is not None else _followup_eligible()
    delta = abs(current_fragility - previous_fragility)
    if delta >= fragility_shift_threshold:
        direction = "increased" if current_fragility > previous_fragility else "decreased"
        status: GardeningStatus = "fragility_shift"
        detail = (
            f"fragility {direction} by {delta:.3f} "
            f"(prev={previous_fragility:.3f}, curr={current_fragility:.3f})"
        )
        needs_followup = _followup
    else:
        status = "healthy"
        detail = f"fragility shift {delta:.3f} within threshold ({fragility_shift_threshold:.3f})"
        needs_followup = False

    return CruxGardeningResult(
        crux_id=entry.crux_id,
        status=status,
        detail=detail,
        previous_fragility=previous_fragility,
        current_fragility=current_fragility,
        needs_followup=needs_followup,
    )


def run_gardening_pass(
    resolved_receipts: list[CruxReceipt],
    outstanding_entries: list[CruxEntry],
    *,
    claim_results: dict[str, ClaimResult] | None = None,
    coherence_issues: list[CoherenceIssue] | None = None,
    fragility_scores: dict[str, tuple[float | None, float | None]] | None = None,
    fragility_shift_threshold: float = DEFAULT_FRAGILITY_SHIFT_THRESHOLD,
    followup_enabled: bool | None = None,
    now: datetime | None = None,
) -> GardeningReport:
    """Run a full gardening pass and return a deterministic :class:`GardeningReport`.

    Parameters
    ----------
    resolved_receipts:
        CruxReceipts from crux-finder debate runs that have been resolved.
    outstanding_entries:
        CruxEntry objects for cruxes still under deliberation.
    claim_results:
        Optional mapping of claim_id → ClaimResult from a DIC-14 verification run.
    coherence_issues:
        Optional list of CoherenceIssue from a DIC-26 coherence scan.
    fragility_scores:
        Optional mapping of crux_id → (previous_fragility, current_fragility) from DIC-25.
    fragility_shift_threshold:
        Minimum absolute delta to surface as a fragility_shift finding.
    now:
        Reference time for the report timestamp; defaults to UTC now.
    """
    generated_at = (now or datetime.now(tz=UTC)).astimezone(UTC).isoformat().replace("+00:00", "Z")
    _followup = followup_enabled if followup_enabled is not None else _followup_eligible()

    resolved_results: list[CruxGardeningResult] = []
    for receipt in resolved_receipts:
        resolved_results.extend(
            garden_resolved_crux(
                receipt,
                claim_results=claim_results,
                coherence_issues=coherence_issues,
                followup_enabled=_followup,
            )
        )

    scores = fragility_scores or {}
    outstanding_results: list[CruxGardeningResult] = [
        garden_outstanding_crux(
            entry,
            previous_fragility=scores.get(entry.crux_id, (None, None))[0],
            current_fragility=scores.get(entry.crux_id, (None, None))[1],
            fragility_shift_threshold=fragility_shift_threshold,
            followup_enabled=_followup,
        )
        for entry in outstanding_entries
    ]

    all_results = resolved_results + outstanding_results
    summary: dict[str, int] = {
        "healthy": 0,
        "stale_evidence": 0,
        "new_contradiction": 0,
        "fragility_shift": 0,
        "needs_followup": 0,
    }
    for r in all_results:
        if r.status in summary:
            summary[r.status] += 1
        if r.needs_followup:
            summary["needs_followup"] += 1

    return GardeningReport(
        generated_at=generated_at,
        resolved_results=resolved_results,
        outstanding_results=outstanding_results,
        summary=summary,
    )


__all__ = [
    "DEFAULT_FRAGILITY_SHIFT_THRESHOLD",
    "CruxGardeningResult",
    "GardeningReport",
    "crux_gardening_enabled",
    "enable_crux_gardening",
    "garden_outstanding_crux",
    "garden_resolved_crux",
    "run_gardening_pass",
]
