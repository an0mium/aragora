"""Receipt persistence and founder-facing rendering for issue triage.

Per the calibration contract, every model call writes an auditable receipt
containing prompt, model id, raw response, parsed verdict, confidence,
cost, latency, and aggregation rationale. This is the receipt-equivalent
artifact Codex required when bypassing Arena for speed.

Three outputs per run:
- ``receipts.jsonl``: one ``IssueDebateReceipt`` per line (audit log).
- ``report.md``: human-readable cards designed to TEACH the founder how
  the panel reached each verdict (not just label it).
- ``summary.json``: run-level summary with verdict + confidence-class
  distributions.

The markdown renderer is deliberately founder-friendly: each card carries
an evidence summary, per-model verdicts with rationale and evidence
anchors, dissent display, and a structured founder-facing recommendation
(action / safety / what-to-inspect / refined-title-body / consolidate-with).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

RECEIPT_SCHEMA_VERSION = "triage-receipt/1.1"


@dataclass
class IssueDebateReceipt:
    """Audit artifact for a single multi-model issue evaluation.

    Schema 1.1 adds founder-facing fields (confidence_class +
    recommendation) and per-model rationale enrichment without breaking
    1.0 readers: every new field has a default.
    """

    issue_number: int
    issue_title: str
    issue_url: str
    issue_author: str
    is_automation_generated: bool

    panel: list[str]
    prompt: str

    per_model: list[dict[str, Any]]
    aggregate_verdict: str
    aggregate_confidence: float
    aggregate_consensus: str
    aggregation_rationale: str
    automation_value: str
    suggested_action: str

    evidence: dict[str, Any]

    started_at: str
    finished_at: str
    cost_usd: float
    latency_seconds: float
    confidence_class: str = "needs-spot-check"
    recommendation: dict[str, Any] | None = None
    schema_version: str = RECEIPT_SCHEMA_VERSION

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


def write_jsonl_receipt(path: Path, receipt: IssueDebateReceipt) -> Path:
    """Append a single receipt to ``path`` (atomic per-line append)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(receipt.to_jsonl_line())
        fh.write("\n")
    return path


HOW_TO_REVIEW_GUIDE = """## How to review this report

Each card below describes one GitHub issue evaluated by a heterogeneous
panel of frontier models. The goal of this report is to TEACH you how
the panel reached its verdict so you can trust or challenge it -- not
to replace your judgement.

Read each card top-to-bottom:

1. **Evidence summary** -- what the panel actually saw (body, labels,
   referenced files in HEAD, related PRs, duplicate candidates). This
   is the ground truth the panel was given.
2. **Per-model verdicts** -- each frontier model's verdict, confidence,
   and rationale, with the specific evidence anchors it cited. If
   models disagree, the dissent block explains why.
3. **Founder-facing recommendation** -- a single structured action
   distilled from the panel: action, why it is safe (or not), what to
   inspect to double-check, and any refined title / body / merge
   target.
4. **Confidence class** -- the panel's own assessment of how risky it
   is to act on the verdict without human review:
   - `easy-call` -- evidence unambiguous, safe to act.
   - `needs-spot-check` -- act after a quick human glance.
   - `do-not-act-without-human` -- ambiguity, dissent, or low
     confidence; require explicit human review.

V1 calibration outputs ARTIFACTS ONLY: no comments, no labels, no
closures, no automation pauses. Closing remains a founder action.

If a verdict is wrong: open the JSONL receipt for that issue, read the
raw model responses, and adjust ``PANEL_PROMPT_RUBRIC`` or the evidence
gathering accordingly. Then re-run that specific issue with
``--issues <N>``.
"""


def write_markdown_report(
    path: Path,
    receipts: Iterable[IssueDebateReceipt],
    *,
    summary_header: str | None = None,
) -> Path:
    """Render a founder-friendly markdown report.

    Layout:
    - Optional summary header (e.g. repo + panel + budget line).
    - Title.
    - Top-of-report ``how to review`` guide.
    - Verdict distribution, confidence-class distribution, automation
      cross-tab.
    - Per-issue cards grouped by verdict, then ordered so easy-call
      cases appear before do-not-act-without-human ones inside each
      group.
    """
    receipts_list = list(receipts)
    grouped: dict[str, list[IssueDebateReceipt]] = {}
    confidence_breakdown: dict[str, int] = {}
    automation_breakdown: dict[str, int] = {}
    for receipt in receipts_list:
        grouped.setdefault(receipt.aggregate_verdict, []).append(receipt)
        confidence_breakdown[receipt.confidence_class] = (
            confidence_breakdown.get(receipt.confidence_class, 0) + 1
        )
        automation_breakdown[receipt.automation_value] = (
            automation_breakdown.get(receipt.automation_value, 0) + 1
        )

    lines: list[str] = []
    if summary_header:
        lines.append(summary_header)
        lines.append("")
    lines.append("# Issue Triage Calibration Report")
    lines.append("")
    lines.append(f"Total issues evaluated: **{len(receipts_list)}**")
    lines.append("")
    lines.append(HOW_TO_REVIEW_GUIDE)
    lines.append("")
    lines.append("## Verdict distribution")
    for verdict in sorted(grouped):
        lines.append(f"- `{verdict}`: {len(grouped[verdict])}")
    lines.append("")
    lines.append("## Confidence-class distribution")
    for cls in ("easy-call", "needs-spot-check", "do-not-act-without-human"):
        if cls in confidence_breakdown:
            lines.append(f"- `{cls}`: {confidence_breakdown[cls]}")
    lines.append("")
    lines.append("## Automation-value cross-tab")
    for value in sorted(automation_breakdown):
        lines.append(f"- `{value}`: {automation_breakdown[value]}")
    lines.append("")
    lines.append(
        "> `automation_value=valuable` is an EXPLICIT POSITIVE outcome. "
        "Automation-origin is not a strike against an issue."
    )
    lines.append("")

    confidence_rank = {
        "easy-call": 0,
        "needs-spot-check": 1,
        "do-not-act-without-human": 2,
    }
    for verdict in sorted(grouped):
        lines.append(f"## Verdict: `{verdict}`")
        lines.append("")
        ordered = sorted(
            grouped[verdict],
            key=lambda r: (confidence_rank.get(r.confidence_class, 99), r.issue_number),
        )
        for receipt in ordered:
            lines.append(_render_card(receipt))
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _render_card(receipt: IssueDebateReceipt) -> str:
    """Render a single founder-facing card for an evaluated issue."""
    auto_tag = (
        f" [automation-{receipt.automation_value}]" if receipt.is_automation_generated else ""
    )
    rec = receipt.recommendation or {}

    body_excerpt = ""
    issue_section = (receipt.evidence or {}).get("issue") or {}
    body = (issue_section.get("body") or "").strip()
    if body:
        snippet = body.splitlines()
        joined = " ".join(line.strip() for line in snippet if line.strip())
        if len(joined) > 320:
            joined = joined[:320] + "..."
        body_excerpt = joined

    refs = receipt.evidence.get("referenced_files") or []
    broken_refs = [r for r in refs if not r.get("exists_in_head")]
    related = receipt.evidence.get("referenced_issues") or []
    dups = receipt.evidence.get("duplicate_candidates") or []

    lines: list[str] = [
        f"### #{receipt.issue_number}: {receipt.issue_title}{auto_tag}",
        "",
        f"- URL: {receipt.issue_url}",
        f"- Author: `{receipt.issue_author}`"
        + (" (automation-origin)" if receipt.is_automation_generated else ""),
        f"- Labels: {', '.join((issue_section.get('labels') or [])) or '(none)'}",
        f"- Confidence class: **`{receipt.confidence_class}`** "
        f"(panel confidence {receipt.aggregate_confidence:.2f}, "
        f"consensus `{receipt.aggregate_consensus}`)",
        "",
        "**Evidence summary**",
    ]
    if body_excerpt:
        lines.append(f"- Body excerpt: {body_excerpt}")
    else:
        lines.append("- Body: (empty)")
    if refs:
        lines.append(
            f"- File references: {len(refs)} mentioned, {len(broken_refs)} missing in HEAD"
        )
        for ref in broken_refs[:5]:
            lines.append(f"  - MISSING in HEAD: `{ref.get('path')}`")
    if related:
        lines.append(f"- Related issues/PRs: {len(related)}")
        for rel in related[:5]:
            lines.append(
                f"  - #{rel.get('number')} state=`{rel.get('state', '?')}` "
                f"title: {(rel.get('title') or '')[:80]}"
            )
    if dups:
        lines.append(f"- Duplicate candidates (Jaccard >= 0.40): {len(dups)}")
        for dup in dups[:5]:
            lines.append(
                f"  - #{dup.get('number')} similarity={dup.get('similarity')} "
                f"title: {(dup.get('title') or '')[:80]}"
            )

    lines.append("")
    lines.append("**Per-model verdicts**")
    for pm in receipt.per_model:
        model_id = pm.get("model_id", "?")
        if pm.get("error"):
            lines.append(f"- `{model_id}`: ERROR: {pm.get('error')}")
            continue
        lines.append(
            f"- `{model_id}` -> **{pm.get('verdict')}** "
            f"(conf {pm.get('confidence', 0.0):.2f}, "
            f"class `{pm.get('confidence_class', '?')}`)"
        )
        if pm.get("rationale"):
            lines.append(f"  - Rationale: {pm['rationale']}")
        evidence_used = pm.get("evidence_used") or []
        if evidence_used:
            lines.append(f"  - Evidence anchors: {', '.join(evidence_used)}")
        if pm.get("what_to_inspect"):
            lines.append(f"  - What to inspect: {pm['what_to_inspect']}")
        if pm.get("safety_note"):
            lines.append(f"  - Safety: {pm['safety_note']}")
        if pm.get("refined_title"):
            lines.append(f"  - Refined title: {pm['refined_title']}")
        if pm.get("refined_body_outline"):
            lines.append(f"  - Refined body outline: {pm['refined_body_outline']}")
        if pm.get("consolidate_with"):
            lines.append(f"  - Consolidate with: #{pm['consolidate_with']}")

    if receipt.aggregate_consensus in ("split", "unclear"):
        lines.append("")
        lines.append("**Dissent**")
        lines.append(f"- {receipt.aggregation_rationale}")

    lines.append("")
    lines.append("**Founder-facing recommendation**")
    action = rec.get("action") or receipt.suggested_action or "(no action provided)"
    safety = rec.get("safety") or ""
    inspect = rec.get("inspect") or ""
    lines.append(f"- **Action:** {action}")
    if safety:
        lines.append(f"- **Why safe / not safe:** {safety}")
    if inspect:
        lines.append(f"- **What to inspect:** {inspect}")
    if rec.get("refined_title"):
        lines.append(f"- **Suggested refined title:** {rec['refined_title']}")
    if rec.get("refined_body_outline"):
        lines.append("- **Suggested refined body outline:**")
        for body_line in str(rec["refined_body_outline"]).splitlines():
            if body_line.strip():
                lines.append(f"  - {body_line.strip().lstrip('-').strip()}")
    if rec.get("consolidate_with"):
        lines.append(f"- **Consolidate into:** #{rec['consolidate_with']}")

    if receipt.notes:
        lines.append("")
        lines.append("**Notes**")
        for note in receipt.notes:
            lines.append(f"- {note}")
    return "\n".join(lines)
