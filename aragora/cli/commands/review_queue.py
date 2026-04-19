"""Read-only PR review queue + advisory packets (Phase 2a of #6279).

Phase 2a is **strictly read-only**: builds a prioritized queue of open PRs and
generates advisory packets for individual PRs. The packet is explicitly
advisory — it does not approve, block, or otherwise settle a PR. Settlement
remains a human action via GitHub UI or ``gh`` directly.

Out of scope for Phase 2a (intentionally not implemented):

- ``review-queue run`` (interactive review session)
- ``review-queue act --approve|--request-changes|--defer`` (settlement actions)
- ``review-queue digest`` (rolled-up activity report)
- Any GitHub write API call

See docs/plans/2026-04-19-batched-pr-review-triage.md for the full design.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

UTC = timezone.utc

# Lane classification thresholds and risk-path catalog.
LARGE_DIFF_THRESHOLD = 500  # additions + deletions, beyond which "needs_human_attention"
HIGH_RISK_PATHS: tuple[str, ...] = (
    "CLAUDE.md",
    "aragora/__init__.py",
    ".env",
    "scripts/nomic_loop.py",
)
HIGH_RISK_PREFIXES: tuple[str, ...] = (
    "aragora/security/",
    "aragora/auth/",
    "aragora/blockchain/",
    "aragora/rbac/",
    "scripts/auto_revert",
    ".github/workflows/",
)
PARKED_LABELS: tuple[str, ...] = ("stale", "do-not-merge", "wip", "blocked")

LANE_ORDER: dict[str, int] = {
    "ready_now": 0,
    "needs_attention": 1,
    "repairable": 2,
    "parked": 3,
}

ADVISORY_NOTE = (
    "This packet is advisory only. It does not approve or block merge. Human settlement required."
)


@dataclass(slots=True)
class QueueItem:
    """One row in the prioritized review queue."""

    number: int
    title: str
    url: str
    head_sha: str
    author: str
    is_draft: bool
    mergeable: str
    review_decision: str
    labels: list[str]
    additions: int
    deletions: int
    changed_files: int
    checks_summary: str
    lane: str
    lane_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewPacket:
    """Advisory packet for one PR. NEVER counts as a GitHub approval."""

    pr_number: int
    title: str
    url: str
    head_sha: str
    author: str
    is_draft: bool
    additions: int
    deletions: int
    changed_files: int
    touched_subsystems: list[str]
    high_risk_paths_touched: list[str]
    checks_summary: str
    risk_flags: list[str]
    machine_recommendation: str
    machine_recommendation_reason: str
    generated_at: str
    advisory_only: bool = True
    settlement_note: str = ADVISORY_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Parser registration ---------------------------------------------------


def add_review_queue_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register 'review-queue' with read-only build/packet sub-actions."""
    parser = subparsers.add_parser(
        "review-queue",
        help="Read-only PR review queue + advisory packets (Phase 2a)",
        description=(
            "Build a prioritized queue of open PRs ready for human review, or\n"
            "generate an advisory packet for one PR.\n\n"
            "Phase 2a is strictly read-only: no approve/request-changes/defer\n"
            "actions, no GitHub writes. See\n"
            "docs/plans/2026-04-19-batched-pr-review-triage.md for the design."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="review_queue_command")

    build_p = sub.add_parser("build", help="Build prioritized review queue from open PRs")
    build_p.add_argument("--limit", type=int, default=100, help="Max PRs to fetch (default: 100)")
    build_p.add_argument(
        "--ready-only",
        action="store_true",
        help="Show only ready_now lane",
    )
    build_p.add_argument(
        "--include-parked",
        action="store_true",
        help="Include parked lane (off by default)",
    )
    build_p.add_argument("--json", action="store_true", help="Output as JSON")

    packet_p = sub.add_parser("packet", help="Generate advisory review packet for one PR")
    packet_p.add_argument("pr", help="PR number")
    packet_p.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override (owner/name). Defaults to current repo context.",
    )
    packet_p.add_argument("--json", action="store_true", help="Output as JSON")

    parser.set_defaults(func=cmd_review_queue)


def cmd_review_queue(args: argparse.Namespace) -> int:
    """Dispatch review-queue subcommands."""
    command = getattr(args, "review_queue_command", None)
    if command == "build":
        return _cmd_build(args)
    if command == "packet":
        return _cmd_packet(args)
    print(
        "Usage: aragora review-queue {build,packet} [...]\n"
        "Phase 2a is read-only. Run 'aragora review-queue build --help' for options.",
        file=sys.stderr,
    )
    return 2


# --- Subcommand entry points -----------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    try:
        items = _build_queue(limit=args.limit)
    except _GhError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    items = _filter_lanes(
        items,
        ready_only=bool(getattr(args, "ready_only", False)),
        include_parked=bool(getattr(args, "include_parked", False)),
    )
    if getattr(args, "json", False):
        print(json.dumps([item.to_dict() for item in items], indent=2))
    else:
        _render_table(items)
    return 0


def _cmd_packet(args: argparse.Namespace) -> int:
    try:
        packet = _build_packet(args.pr, repo_override=args.repo)
    except _GhError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        _render_packet(packet)
    return 0


# --- Internals: gh shell, classification, packet building ------------------


class _GhError(RuntimeError):
    """Raised when a 'gh' invocation fails or returns malformed JSON."""


def _gh_json(args: list[str]) -> Any:
    """Run a 'gh' command and parse JSON output. Returns None for empty stdout."""
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "no stderr"
        raise _GhError(f"gh {' '.join(args)} failed: {stderr}")
    out = proc.stdout.strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as exc:
        raise _GhError(f"gh {' '.join(args)} returned malformed JSON: {exc}") from exc


def _build_queue(*, limit: int) -> list[QueueItem]:
    fields = ",".join(
        [
            "number",
            "title",
            "url",
            "headRefName",
            "headRefOid",
            "isDraft",
            "mergeable",
            "reviewDecision",
            "labels",
            "author",
            "additions",
            "deletions",
            "changedFiles",
            "statusCheckRollup",
        ]
    )
    raw = _gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            str(limit),
            "--json",
            fields,
        ]
    )
    items: list[QueueItem] = []
    for pr in raw or []:
        if not isinstance(pr, dict):
            continue
        items.append(_classify_pr(pr))
    items.sort(key=lambda it: (LANE_ORDER.get(it.lane, 99), -it.number))
    return items


def _classify_pr(pr: dict[str, Any]) -> QueueItem:
    """Assign one PR to a lane based on draft/checks/diff/labels signals."""
    number = int(pr.get("number", 0) or 0)
    title = str(pr.get("title", "")).strip()
    url = str(pr.get("url", "")).strip()
    head_sha = str(pr.get("headRefOid", "")).strip()
    is_draft = bool(pr.get("isDraft", False))
    mergeable = str(pr.get("mergeable", "")).strip().upper()
    review_decision = str(pr.get("reviewDecision", "")).strip().upper()
    labels = [
        str(lab.get("name", "")).strip()
        for lab in (pr.get("labels") or [])
        if isinstance(lab, dict) and lab.get("name")
    ]
    author = ""
    author_payload = pr.get("author")
    if isinstance(author_payload, dict):
        author = str(author_payload.get("login", "")).strip()
    additions = int(pr.get("additions", 0) or 0)
    deletions = int(pr.get("deletions", 0) or 0)
    changed_files = int(pr.get("changedFiles", 0) or 0)
    checks_summary, has_failures, has_pending = _summarize_checks(pr.get("statusCheckRollup") or [])

    parked_label_hits = [lab for lab in labels if lab in PARKED_LABELS]

    if is_draft:
        lane, reason = "parked", "draft PR"
    elif parked_label_hits:
        lane, reason = "parked", f"label={','.join(parked_label_hits)}"
    elif mergeable == "CONFLICTING":
        lane, reason = "parked", "merge conflict"
    elif has_failures:
        lane, reason = "repairable", checks_summary
    elif has_pending:
        lane, reason = "needs_attention", f"checks pending ({checks_summary})"
    elif additions + deletions > LARGE_DIFF_THRESHOLD:
        lane, reason = "needs_attention", f"large diff (+{additions}/-{deletions})"
    elif mergeable in ("MERGEABLE", "UNKNOWN", ""):
        lane, reason = "ready_now", checks_summary or "all green"
    else:
        lane, reason = "needs_attention", f"mergeable={mergeable}"

    return QueueItem(
        number=number,
        title=title,
        url=url,
        head_sha=head_sha,
        author=author,
        is_draft=is_draft,
        mergeable=mergeable,
        review_decision=review_decision,
        labels=labels,
        additions=additions,
        deletions=deletions,
        changed_files=changed_files,
        checks_summary=checks_summary,
        lane=lane,
        lane_reason=reason,
    )


def _summarize_checks(checks: list) -> tuple[str, bool, bool]:
    """Return ``(summary, has_failures, has_pending)`` for a statusCheckRollup."""
    success = failure = pending = 0
    for check in checks:
        if not isinstance(check, dict):
            continue
        status = str(check.get("status") or check.get("state") or "").upper()
        conclusion = str(check.get("conclusion") or "").upper()
        # Status-context rollups use ``state`` without a separate conclusion.
        # Normalize those terminal states into the same summary buckets.
        if not conclusion and status in {
            "SUCCESS",
            "FAILURE",
            "TIMED_OUT",
            "ACTION_REQUIRED",
            "CANCELLED",
            "SKIPPED",
            "NEUTRAL",
            "STALE",
        }:
            conclusion = status
        elif not conclusion and status in {"ERROR", "FAILED"}:
            conclusion = "FAILURE"
        if conclusion == "SUCCESS":
            success += 1
        elif conclusion in ("FAILURE", "TIMED_OUT", "ACTION_REQUIRED"):
            failure += 1
        elif conclusion in ("CANCELLED", "SKIPPED", "NEUTRAL", "STALE"):
            # Treat skipped/cancelled as not-meaningful for the summary; they
            # are correct gating behavior in this repo (see docs/CI_LANES.md).
            continue
        elif status in ("IN_PROGRESS", "QUEUED", "PENDING", "EXPECTED") or not conclusion:
            pending += 1
    total = success + failure + pending
    if failure > 0:
        return (f"{failure} failing / {total} total", True, pending > 0)
    if pending > 0:
        return (f"{pending} pending / {total} total", False, True)
    if success > 0:
        return (f"{success}/{total} green", False, False)
    return ("no checks", False, False)


def _filter_lanes(
    items: list[QueueItem],
    *,
    ready_only: bool,
    include_parked: bool,
) -> list[QueueItem]:
    if ready_only:
        return [it for it in items if it.lane == "ready_now"]
    if not include_parked:
        return [it for it in items if it.lane != "parked"]
    return items


def _build_packet(pr_ref: str, *, repo_override: str | None) -> ReviewPacket:
    number = _parse_pr_number(pr_ref)
    fields = ",".join(
        [
            "number",
            "title",
            "url",
            "headRefName",
            "headRefOid",
            "isDraft",
            "mergeable",
            "reviewDecision",
            "labels",
            "author",
            "additions",
            "deletions",
            "changedFiles",
            "statusCheckRollup",
            "files",
        ]
    )
    args = ["pr", "view", str(number), "--json", fields]
    if repo_override:
        args.extend(["--repo", repo_override])
    pr = _gh_json(args)
    if pr is None or not isinstance(pr, dict):
        raise _GhError(f"PR #{number} not found")

    files: list[str] = []
    for item in pr.get("files") or []:
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            if path:
                files.append(path)
    labels = [
        str(lab.get("name", "")).strip()
        for lab in (pr.get("labels") or [])
        if isinstance(lab, dict) and lab.get("name")
    ]
    parked_label_hits = [lab for lab in labels if lab in PARKED_LABELS]
    touched = sorted({_subsystem_for(p) for p in files})
    high_risk = [p for p in files if _is_high_risk_path(p)]
    checks_summary, has_failures, has_pending = _summarize_checks(pr.get("statusCheckRollup") or [])
    additions = int(pr.get("additions", 0) or 0)
    deletions = int(pr.get("deletions", 0) or 0)
    is_draft = bool(pr.get("isDraft", False))
    mergeable = str(pr.get("mergeable", "")).strip().upper()

    risk_flags: list[str] = []
    if is_draft:
        risk_flags.append("draft PR")
    if parked_label_hits:
        risk_flags.append(f"parked label ({','.join(parked_label_hits)})")
    if high_risk:
        sample = ", ".join(high_risk[:5])
        more = "" if len(high_risk) <= 5 else f" (+{len(high_risk) - 5} more)"
        risk_flags.append(f"touches high-risk paths: {sample}{more}")
    if additions + deletions > LARGE_DIFF_THRESHOLD:
        risk_flags.append(f"large diff (+{additions}/-{deletions})")
    if mergeable == "CONFLICTING":
        risk_flags.append("merge conflict")
    if has_failures:
        risk_flags.append(f"checks failing ({checks_summary})")

    if has_failures or mergeable == "CONFLICTING":
        recommendation = "repair_first"
        recommendation_reason = "checks failing or merge conflict — fix before review"
    elif is_draft:
        recommendation = "needs_human_attention"
        recommendation_reason = "draft PR — keep parked until it is ready for review"
    elif parked_label_hits:
        recommendation = "needs_human_attention"
        recommendation_reason = (
            f"parked label present ({','.join(parked_label_hits)}) — keep parked until cleared"
        )
    elif high_risk or additions + deletions > LARGE_DIFF_THRESHOLD:
        recommendation = "needs_human_attention"
        recommendation_reason = "high-risk paths touched or large diff — human should read it"
    elif has_pending:
        recommendation = "needs_human_attention"
        recommendation_reason = "checks still pending — wait for completion"
    else:
        recommendation = "approve_candidate"
        recommendation_reason = "all green, bounded diff, no high-risk paths"

    author = ""
    author_payload = pr.get("author")
    if isinstance(author_payload, dict):
        author = str(author_payload.get("login", "")).strip()

    return ReviewPacket(
        pr_number=number,
        title=str(pr.get("title", "")).strip(),
        url=str(pr.get("url", "")).strip(),
        head_sha=str(pr.get("headRefOid", "")).strip(),
        author=author,
        is_draft=is_draft,
        additions=additions,
        deletions=deletions,
        changed_files=int(pr.get("changedFiles", 0) or 0),
        touched_subsystems=touched,
        high_risk_paths_touched=high_risk,
        checks_summary=checks_summary,
        risk_flags=risk_flags,
        machine_recommendation=recommendation,
        machine_recommendation_reason=recommendation_reason,
        generated_at=datetime.now(UTC).isoformat(),
    )


def _subsystem_for(path: str) -> str:
    """Map a file path to a coarse subsystem label for risk grouping."""
    parts = path.split("/")
    if not parts:
        return "(root)"
    top = parts[0]
    if top in ("aragora", "tests") and len(parts) >= 2:
        return f"{top}/{parts[1]}"
    if top in ("docs", "scripts", "sdk", "benchmarks", ".github"):
        return top
    return top


def _is_high_risk_path(path: str) -> bool:
    if path in HIGH_RISK_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in HIGH_RISK_PREFIXES)


def _parse_pr_number(pr_ref: str) -> int:
    text = str(pr_ref).strip()
    if "/" in text:
        text = text.rstrip("/").split("/")[-1]
    if text.startswith("#"):
        text = text[1:]
    try:
        return int(text)
    except ValueError as exc:
        raise _GhError(f"invalid PR ref: {pr_ref!r}") from exc


# --- Rendering -------------------------------------------------------------


def _render_table(items: list[QueueItem]) -> None:
    if not items:
        print("(no PRs in scope)")
        return
    counts: dict[str, int] = {}
    for item in items:
        counts[item.lane] = counts.get(item.lane, 0) + 1
    lane_summary = ", ".join(f"{lane}={counts.get(lane, 0)}" for lane in LANE_ORDER)
    print(f"Review queue ({len(items)} PRs): {lane_summary}")
    print()
    current_lane = ""
    for item in items:
        if item.lane != current_lane:
            current_lane = item.lane
            print(f"== {item.lane} ==")
        title_clip = item.title[:70]
        print(
            f"  #{item.number:>5}  {item.checks_summary:>20}  "
            f"+{item.additions:>5}/-{item.deletions:<5}  {title_clip}"
        )
        print(f"        {item.url}  [{item.lane_reason}]")
    print()
    print(
        "Note: this queue is advisory only. Settlement (approve/request-changes/defer) "
        "requires human action via GitHub UI or `gh pr review` / `gh pr merge`."
    )


def _render_packet(packet: ReviewPacket) -> None:
    print(f"# Advisory review packet — PR #{packet.pr_number}")
    print(f"# {packet.title}")
    print(f"# {packet.url}")
    print()
    print(f"head SHA:        {packet.head_sha}")
    print(f"author:          {packet.author}")
    print(f"draft:           {packet.is_draft}")
    print(
        f"diff:            +{packet.additions}/-{packet.deletions} "
        f"across {packet.changed_files} files"
    )
    print(f"checks:          {packet.checks_summary}")
    print()
    if packet.touched_subsystems:
        print("touched subsystems:")
        for sub in packet.touched_subsystems:
            print(f"  - {sub}")
        print()
    if packet.high_risk_paths_touched:
        print("HIGH-RISK PATHS TOUCHED:")
        for path in packet.high_risk_paths_touched:
            print(f"  - {path}")
        print()
    if packet.risk_flags:
        print("risk flags:")
        for flag in packet.risk_flags:
            print(f"  - {flag}")
        print()
    print(f"machine recommendation: {packet.machine_recommendation}")
    print(f"  reason: {packet.machine_recommendation_reason}")
    print()
    print(f"generated at: {packet.generated_at}")
    print()
    print(f"-- {packet.settlement_note}")
