"""Compound H1 multi-gate readiness aggregator (advisory-only).

The H1 epic has four sub-gates: H1-01 (rev-4 promotion), H1-02
(scorecard publication), H1-03 (sanitation gate), H1-04 (autonomy
ledger self-heal). Each has its own contract document under
``docs/status/H1_*_CONTRACT.md`` and its own readiness check.

For an operator running a release decision, *all four* must be in a
known-good state. Today the operator checks each by hand.

This module provides a pure aggregator that ingests the per-gate
status records and returns one ``MultiGateReadiness`` value plus a
deterministic Markdown rendering. It makes **no GitHub or filesystem
calls itself** — the caller passes in already-loaded dicts. That keeps
the aggregator fast, deterministic, and unit-testable offline.

The intent is that the renderer in
``scripts/render_h1_multi_gate_readiness.py`` reads the per-gate
artifacts (the rev-4 readiness JSON, the H1-02 scorecard JSON, the
sanitizer test surface, the ledger self-heal contract) and feeds them
into ``aggregate_readiness`` here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal

GateStatus = Literal["ready", "advisory_in_progress", "in_progress", "blocked", "unknown"]

# The four H1 sub-gates in canonical roadmap order.
H1_GATES: Final[tuple[str, ...]] = ("H1-01", "H1-02", "H1-03", "H1-04")


@dataclass(frozen=True, slots=True)
class GateInput:
    """One H1 sub-gate's per-gate readiness signal.

    Attributes:
        gate_id: One of ``H1_GATES``.
        status: ``ready`` (canonical contract IN PLACE),
            ``advisory_in_progress`` (advisory deliverable shipped but
            not promoted), ``in_progress`` (work ongoing),
            ``blocked`` (blocker known), ``unknown`` (no signal yet).
        contract_doc: Path to the contract document, if any.
        evidence: Free-form mapping; preserved as-is in the output.
    """

    gate_id: str
    status: GateStatus
    contract_doc: str | None = None
    evidence: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MultiGateReadiness:
    """Aggregated H1 readiness across all four sub-gates."""

    overall_status: GateStatus
    ready_count: int
    advisory_count: int
    blocked_count: int
    in_progress_count: int
    unknown_count: int
    per_gate: tuple[GateInput, ...]
    next_action: str


def aggregate_readiness(inputs: list[GateInput]) -> MultiGateReadiness:
    """Combine per-gate inputs into one readiness verdict.

    Rules:
        - All four gates ``ready`` → overall ``ready``.
        - Any gate ``blocked`` → overall ``blocked`` (binding).
        - Otherwise, if any gate is ``in_progress`` → overall
          ``in_progress``.
        - Otherwise (mix of ready / advisory_in_progress) → overall
          ``advisory_in_progress``.
        - Otherwise (any ``unknown``) → overall ``unknown``.

    The aggregator is order-independent and pure: same inputs always
    produce the same output. Missing gates count as ``unknown``.
    """
    by_id: dict[str, GateInput] = {gi.gate_id: gi for gi in inputs}
    normalized: list[GateInput] = []
    for gid in H1_GATES:
        if gid in by_id:
            normalized.append(by_id[gid])
        else:
            normalized.append(
                GateInput(gate_id=gid, status="unknown", contract_doc=None, evidence={})
            )

    statuses = [gi.status for gi in normalized]
    ready_count = statuses.count("ready")
    advisory_count = statuses.count("advisory_in_progress")
    blocked_count = statuses.count("blocked")
    in_progress_count = statuses.count("in_progress")
    unknown_count = statuses.count("unknown")

    overall: GateStatus
    next_action: str
    if blocked_count > 0:
        overall = "blocked"
        blocked_ids = [gi.gate_id for gi in normalized if gi.status == "blocked"]
        next_action = f"Resolve blocker(s) on {', '.join(blocked_ids)} before any other gate work."
    elif ready_count == len(H1_GATES):
        overall = "ready"
        next_action = "All four H1 sub-gates ready. H1 epic may be marked complete."
    elif in_progress_count > 0:
        overall = "in_progress"
        in_prog_ids = [gi.gate_id for gi in normalized if gi.status == "in_progress"]
        next_action = f"Continue work on {', '.join(in_prog_ids)}."
    elif advisory_count > 0:
        overall = "advisory_in_progress"
        advisory_ids = [gi.gate_id for gi in normalized if gi.status == "advisory_in_progress"]
        next_action = (
            f"Promote {', '.join(advisory_ids)} from advisory → canonical to graduate the H1 epic."
        )
    else:
        # No blockers, not all ready, no in-progress, no advisory →
        # only readys + unknowns, or all unknowns.
        overall = "unknown"
        unknown_ids = [gi.gate_id for gi in normalized if gi.status == "unknown"]
        next_action = (
            f"Run per-gate readiness checks for {', '.join(unknown_ids)} before aggregating."
        )

    return MultiGateReadiness(
        overall_status=overall,
        ready_count=ready_count,
        advisory_count=advisory_count,
        blocked_count=blocked_count,
        in_progress_count=in_progress_count,
        unknown_count=unknown_count,
        per_gate=tuple(normalized),
        next_action=next_action,
    )


_STATUS_BADGE: Final[dict[GateStatus, str]] = {
    "ready": "ready",
    "advisory_in_progress": "advisory",
    "in_progress": "in-progress",
    "blocked": "BLOCKED",
    "unknown": "unknown",
}


def render_markdown(readiness: MultiGateReadiness) -> str:
    """Deterministic Markdown rendering for the readiness verdict.

    The output is stable across calls with the same input — no clocks,
    no ordering ambiguity — so it can be diffed across rounds.
    """
    overall_badge = _STATUS_BADGE[readiness.overall_status]
    lines: list[str] = [
        "# H1 multi-gate readiness",
        "",
        f"**Overall status:** `{overall_badge}`",
        "",
        f"- Ready: **{readiness.ready_count}**/{len(H1_GATES)}",
        f"- Advisory in progress: **{readiness.advisory_count}**",
        f"- In progress: **{readiness.in_progress_count}**",
        f"- Blocked: **{readiness.blocked_count}**",
        f"- Unknown: **{readiness.unknown_count}**",
        "",
        "## Per-gate",
        "",
        "| Gate | Status | Contract |",
        "| --- | --- | --- |",
    ]
    for gi in readiness.per_gate:
        contract = gi.contract_doc or "—"
        lines.append(f"| `{gi.gate_id}` | `{_STATUS_BADGE[gi.status]}` | {contract} |")
    lines.extend(
        [
            "",
            "## Next action",
            "",
            readiness.next_action,
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "GateInput",
    "GateStatus",
    "H1_GATES",
    "MultiGateReadiness",
    "aggregate_readiness",
    "render_markdown",
]
