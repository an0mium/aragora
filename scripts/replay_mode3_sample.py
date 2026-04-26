#!/usr/bin/env python3
"""Replay stored v2.9.0-rc.1 briefs through the post-#6505 rubric.

Closes fix #4 of epic #6505: "Re-derive precision on the 15-brief sample
post-fix." Takes the briefs already on disk under
``.aragora/review-queue/briefs/`` and re-synthesizes each through the new
verdict rules landed in #6506 (severity counts), #6510
(``APPROVE_WITH_FOLLOWUPS`` + severity gate), and #6514 (advocate lens).

No new API spend: the old briefs' ``role_findings`` are the only input.

Scope limitations — honest labeling

The archived briefs predate #6506, so they do NOT carry structured
severity data. This script derives ``findings_severity_counts``
heuristically from each finding's ``finding_text`` using a keyword
classifier. That makes the replay a *plausibility check* on the new
rubric, not a re-scoring with oracle severity labels. Where a real
label is available (manual ``*.severity.json`` sidecar), the script
prefers that over the heuristic — see ``_load_label_override``.

The advocate lens simulation is similarly heuristic: the archived
briefs did not run an advocate slot, so we synthesize a plausible
APPROVE vote from each brief's ``validation_summary`` (tests/ruff/CI
evidence) at a conservative confidence. This is a floor estimate — a
real advocate panelist could argue more forcefully.

Output is a markdown table comparing per-brief:

  - old_verdict              (as stored)
  - severity_counts_estimate (heuristic, from finding_text)
  - new_verdict_sev_gate     (severity gate applied; no advocate)
  - new_verdict_full         (severity gate + simulated advocate)

Plus aggregate rollups. Intended consumer is
``docs/status/2026-04-24-mode3-rc1-calibration-post-fix.md``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
BRIEFS_DIR = REPO_ROOT / ".aragora" / "review-queue" / "briefs"

# --- Severity keyword classifier -----------------------------------------
#
# Applied to each finding's ``finding_text``. A finding can match multiple
# tiers; highest-severity match wins. The vocabulary is tuned for the
# language these panels actually use — not a general severity ontology.

_HIGH_KEYWORDS = (
    # explicit blocker / critical language
    r"\bblocker\b",
    r"\bcritical\b",
    r"\bhard blocker\b",
    r"\bmerge[- ]critical\b",
    r"\bexactly wrong\b",
    # security / correctness failures
    r"\bsecurity\s+vuln",
    r"\bvulnerability\b",
    r"\bdata loss\b",
    r"\bdata corruption\b",
    r"\bsilent(ly)? divergen",
    r"\bsilent(ly)? fail",
    r"\brace condition\b",
    r"\bdeadlock\b",
    # authoritative "must fix" constructions
    r"\bmust\s+be\s+(fixed|addressed)\b",
    r"\bmust\s+not\s+merge\b",
    r"\bshould\s+not\s+merge\b",
)

_MEDIUM_KEYWORDS = (
    r"\bshould\s+be\s+(acknowledged|addressed|extracted|fixed|documented)\b",
    r"\btechnical\s+debt\b",
    r"\bmaintainability\s+debt\b",
    r"\bfragil(e|ity)\b",
    r"\bcoupling\b",
    r"\bscope\s+creep\b",
    r"\bdegrade",
    r"\blong[- ]tail\s+risk",
    r"\bhidden\s+coupling\b",
    r"\blatent\s+debt\b",
    r"\b(un)?coordinated\s+sqlite",
    r"\bdocumentation\s+gaps?\b",
    r"\bnot\s+tested\b",
    r"\buntested\b",
)

_LOW_KEYWORDS = (
    r"\bminor\b",
    r"\bnitpick\b",
    r"\beditorial\b",
    r"\bopportunity\b",
    r"\bconsider(ing)?\b",
    r"\bstyle\b",
    r"\bcosmetic\b",
    r"\bcould\s+(also\s+)?be\b",
    r"\blow[- ]risk\b",
)


def _classify_severity(text: str) -> str:
    """Return ``"high"``, ``"medium"``, or ``"low"`` for one finding.

    The classifier is intentionally conservative-by-default: when no tier
    matches, the finding is labelled ``"low"`` rather than dropped. That
    biases the output toward false-approves (understating blockers), not
    false-blocks (overstating blockers) — the less destructive error mode
    for a calibration exercise.
    """
    lowered = text.lower()
    for pattern in _HIGH_KEYWORDS:
        if re.search(pattern, lowered):
            return "high"
    for pattern in _MEDIUM_KEYWORDS:
        if re.search(pattern, lowered):
            return "medium"
    for pattern in _LOW_KEYWORDS:
        if re.search(pattern, lowered):
            return "low"
    return "low"


# --- Synthetic advocate vote ---------------------------------------------
#
# The archived briefs never ran an advocate slot. Rather than exclude
# advocate from the replay (and so understate the rubric change), we
# synthesize a vote from each brief's ``validation_summary`` field, which
# records the tests/CI evidence that an advocate would lean on.

_APPROVE_EVIDENCE = (
    r"\bpass(ing|ed)?\s+pytest\b",
    r"\bpass(ing|ed)?\s+tests?\b",
    r"\bclean\s+ruff\b",
    r"\bruff\s+clean\b",
    r"\bpre[- ]push\s+hooks?\s+pass",
    r"\bgreen\b",
    r"\bno\s+regressions?\b",
    r"\bcovered\s+by\s+tests\b",
    r"\btests?\s+exercise\b",
)

_APPROVE_PENALTIES = (
    r"\bno\s+ci\s+artifact",
    r"\btests?\s+do\s+not\s+exercise\b",
    r"\buntested\b",
    r"\bauthor[- ]asserted\b",
    r"\bevidence\s+strength\s+is\s+low\b",
)


def _advocate_confidence(validation_summary: str) -> float:
    """Simulate a floor advocate confidence from stored validation evidence.

    Returns a scalar in ``[0.0, 1.0]`` that a real advocate could lean on
    to argue APPROVE. Pure heuristic; documented as a floor estimate —
    a real advocate panelist would read the diff + evidence and could
    argue more forcefully than this function ever does.
    """
    lowered = validation_summary.lower()
    hits = sum(1 for p in _APPROVE_EVIDENCE if re.search(p, lowered))
    penalties = sum(1 for p in _APPROVE_PENALTIES if re.search(p, lowered))
    # Start conservative: each piece of positive evidence is worth 0.10,
    # capped at 0.70; penalties subtract 0.20 each.
    score = min(0.70, 0.10 * hits) - 0.20 * penalties
    return max(0.0, round(score, 3))


# --- Verdict rule application --------------------------------------------


OLD_VERDICTS = ("approve_candidate", "repair_first", "needs_human_attention")
NEW_VERDICTS = (
    "approve_candidate",
    "approve_with_followups",
    "repair_first",
    "needs_human_attention",
)


def apply_severity_gate(
    old_recommendation: str,
    severity_counts: Mapping[str, int],
) -> str:
    """Mirror ``aragora.review.builder._apply_severity_gate``.

    Rule: a brief whose primary verdict is ``repair_first`` downgrades
    to ``approve_with_followups`` when the panel produced NO ``high``-
    severity findings. All other verdicts pass through unchanged.
    """
    if old_recommendation != "repair_first":
        return old_recommendation
    if severity_counts.get("high", 0) > 0:
        return "repair_first"
    return "approve_with_followups"


def apply_advocate_rebalance(
    sev_gated_verdict: str,
    panel_weight_against_approve: float,
    advocate_confidence: float,
) -> str:
    """Apply the advocate lens to the severity-gated verdict.

    Weighted-policy intuition: the advocate adds ``advocate_confidence``
    to the APPROVE position's score. If the old brief was
    ``repair_first`` with a modest against-approve weight, adding the
    advocate's approve-weight can flip the vote to approve. For
    ``needs_human_attention`` (tied vote), the advocate can break the
    tie toward approve.

    This function is intentionally monotone: the advocate never pushes
    the verdict in a more blocking direction.
    """
    if sev_gated_verdict == "approve_candidate":
        return sev_gated_verdict
    if advocate_confidence >= panel_weight_against_approve:
        if sev_gated_verdict in ("repair_first", "needs_human_attention"):
            return "approve_candidate"
        if sev_gated_verdict == "approve_with_followups":
            return "approve_candidate"
    return sev_gated_verdict


# --- Per-brief replay -----------------------------------------------------


@dataclass(frozen=True)
class BriefReplay:
    pr_number: int
    head_sha: str
    old_verdict: str
    severity_counts: dict[str, int]
    new_verdict_sev_gate: str
    new_verdict_full: str
    advocate_confidence: float
    panel_weight_against_approve: float
    overall_confidence: float
    role_findings_count: int
    notes: str


def _load_label_override(brief_path: Path) -> Mapping[str, int] | None:
    """Prefer manual severity labels over the keyword heuristic, if present.

    A sidecar at ``<brief>.severity.json`` containing
    ``{"high": N, "medium": N, "low": N}`` lets an operator override the
    heuristic per-brief when a real label exists. Absence returns None
    and the heuristic runs.
    """
    sidecar = brief_path.with_suffix(".severity.json")
    if not sidecar.exists():
        return None
    data = json.loads(sidecar.read_text())
    return {
        "high": int(data.get("high", 0)),
        "medium": int(data.get("medium", 0)),
        "low": int(data.get("low", 0)),
    }


def replay_brief(brief_path: Path) -> BriefReplay:
    data = json.loads(brief_path.read_text())

    role_findings = data.get("role_findings", [])
    label_override = _load_label_override(brief_path)
    if label_override is not None:
        severity_counts = dict(label_override)
        notes = "severity=manual"
    else:
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        for rf in role_findings:
            sev = _classify_severity(rf.get("finding_text", ""))
            severity_counts[sev] += 1
        notes = "severity=heuristic"

    old_verdict = data.get("recommendation", "")
    new_sev_gate = apply_severity_gate(old_verdict, severity_counts)

    # For the full replay we need to know how much confidence the panel
    # put behind NOT-APPROVE. Without stored positions we use the stored
    # ``overall_confidence`` as an upper bound — it's a mean across the
    # panel, and the archived briefs all had unanimous request_changes
    # positions per the synthesizer top_lines, so panel_weight_against
    # ≈ overall_confidence × panel_size. We scale by panel_size to make
    # the advocate comparison per-vote-equivalent.
    overall_confidence = float(data.get("overall_confidence", 0.0))
    panel_size = max(1, len(role_findings))
    panel_weight_against_approve = overall_confidence  # per-vote-equivalent

    advocate_conf = _advocate_confidence(data.get("validation_summary", ""))
    new_full = apply_advocate_rebalance(
        new_sev_gate,
        panel_weight_against_approve=panel_weight_against_approve,
        advocate_confidence=advocate_conf,
    )

    return BriefReplay(
        pr_number=int(data.get("pr_number", 0)),
        head_sha=str(data.get("head_sha", ""))[:12],
        old_verdict=old_verdict,
        severity_counts=severity_counts,
        new_verdict_sev_gate=new_sev_gate,
        new_verdict_full=new_full,
        advocate_confidence=advocate_conf,
        panel_weight_against_approve=round(panel_weight_against_approve, 3),
        overall_confidence=round(overall_confidence, 3),
        role_findings_count=panel_size,
        notes=notes,
    )


def iter_briefs(briefs_dir: Path) -> Iterable[Path]:
    yield from sorted(briefs_dir.glob("pr-*.json"))


def _missing_briefs_error(briefs_dir: Path) -> str:
    message = f"error: briefs directory not found: {briefs_dir}"
    if briefs_dir == BRIEFS_DIR:
        message += (
            "\n"
            "hint: .aragora/ is intentionally gitignored; rerun with "
            "--briefs-dir pointing at a local archived brief directory."
        )
    return message


# --- Reporting ------------------------------------------------------------


def format_markdown(replays: list[BriefReplay]) -> str:
    lines: list[str] = []
    lines.append("## Per-brief replay")
    lines.append("")
    lines.append(
        "| PR | sha | old | sev_counts (h/m/l) | sev_gate_only | full_replay | adv_conf | notes |"
    )
    lines.append(
        "|----|-----|-----|--------------------|---------------|-------------|----------|-------|"
    )
    for r in replays:
        sc = r.severity_counts
        lines.append(
            f"| #{r.pr_number} | `{r.head_sha}` | {r.old_verdict} | "
            f"{sc['high']}/{sc['medium']}/{sc['low']} | "
            f"{r.new_verdict_sev_gate} | {r.new_verdict_full} | "
            f"{r.advocate_confidence} | {r.notes} |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize(replays: list[BriefReplay]) -> str:
    total = len(replays)
    if total == 0:
        return "No briefs replayed."

    def dist(key: str) -> dict[str, int]:
        d: dict[str, int] = {}
        for r in replays:
            v = getattr(r, key)
            d[v] = d.get(v, 0) + 1
        return d

    old_dist = dist("old_verdict")
    sev_dist = dist("new_verdict_sev_gate")
    full_dist = dist("new_verdict_full")

    def fmt(d: dict[str, int]) -> str:
        return ", ".join(f"{k}: {v}/{total}" for k, v in sorted(d.items()))

    out: list[str] = []
    out.append("## Aggregate verdict distribution")
    out.append("")
    out.append(f"- **Old (as stored):** {fmt(old_dist)}")
    out.append(f"- **New (severity gate only):** {fmt(sev_dist)}")
    out.append(f"- **New (severity gate + advocate):** {fmt(full_dist)}")
    out.append("")
    heuristic = sum(1 for r in replays if r.notes == "severity=heuristic")
    manual = total - heuristic
    out.append(f"*Severity source: {heuristic}/{total} heuristic, {manual}/{total} manual labels.*")
    out.append("")
    return "\n".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--briefs-dir",
        type=Path,
        default=BRIEFS_DIR,
        help="Directory containing pr-*.json brief files.",
    )
    p.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to stdout.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.briefs_dir.exists():
        print(_missing_briefs_error(args.briefs_dir), file=sys.stderr)
        return 2

    replays = [replay_brief(p) for p in iter_briefs(args.briefs_dir)]

    if args.format == "json":
        payload = [
            {
                "pr_number": r.pr_number,
                "head_sha": r.head_sha,
                "old_verdict": r.old_verdict,
                "severity_counts": r.severity_counts,
                "new_verdict_sev_gate": r.new_verdict_sev_gate,
                "new_verdict_full": r.new_verdict_full,
                "advocate_confidence": r.advocate_confidence,
                "panel_weight_against_approve": r.panel_weight_against_approve,
                "overall_confidence": r.overall_confidence,
                "role_findings_count": r.role_findings_count,
                "notes": r.notes,
            }
            for r in replays
        ]
        output = json.dumps(payload, indent=2, sort_keys=True)
    else:
        output = summarize(replays) + "\n" + format_markdown(replays)

    if args.out:
        args.out.write_text(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
