"""PR review queue, advisory packets, and human settlement (Phases 2a/2b).

This command keeps machine review advisory-only while making the human
settlement step fast and receipt-backed:

- ``build`` prioritizes open PRs for founder review
- ``packet`` builds the advisory packet for one live PR head
- ``run`` walks a human through approve/request-changes/defer in one loop
- ``act`` performs one explicit human settlement action with freshness checks

Out of scope (intentionally still not implemented):

- ``review-queue digest`` (rolled-up activity report)
- ``merge_arbiter`` enforcement of settlement receipts
- Bot-only merge on green CI
- Any hidden merge path that bypasses explicit human action

See docs/plans/2026-04-19-batched-pr-review-triage.md for the full design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.review.reviewer_output import ReviewerOutput
from aragora.swarm.pr_review_protocol import PRReviewerExecutionFailure, default_pr_review_protocol
from aragora.worktree.fleet import resolve_repo_root

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
REVIEW_QUEUE_ARTIFACT_DIR = ".aragora/review-queue"
REQUEST_CHANGES_REASON_REQUIRED = (
    "request-changes requires a one-line human reason so the repair loop stays bounded."
)
DEFER_REASON_REQUIRED = (
    "defer requires a one-line human reason so the PR does not disappear silently."
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
    base_sha: str
    author: str
    is_draft: bool
    additions: int
    deletions: int
    changed_files: int
    queue_bucket: str
    touched_subsystems: list[str]
    high_risk_paths_touched: list[str]
    validation: list[str]
    checks_summary: str
    risk_flags: list[str]
    machine_recommendation: str
    machine_recommendation_reason: str
    packet_sha: str
    generated_at: str
    protocol: dict[str, Any] = field(default_factory=dict)
    advisory_only: bool = True
    settlement_note: str = ADVISORY_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SettlementReceipt:
    """Persisted human settlement receipt for one PR/head/packet tuple."""

    session_id: str
    reviewed_at: str
    actor: str
    action: str
    reason: str
    pr_number: int
    pr_url: str
    head_sha: str
    base_sha: str
    packet_sha: str
    queue_bucket: str
    machine_recommendation: str
    github_event: str
    elapsed_seconds: float | None = None
    receipt_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Parser registration ---------------------------------------------------


def add_review_queue_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register review-queue build/packet/run/act sub-actions."""
    parser = subparsers.add_parser(
        "review-queue",
        help="PR review queue + advisory packets + human settlement",
        description=(
            "Build a prioritized queue of open PRs ready for human review, or\n"
            "generate an advisory packet for one PR, or settle one PR with an\n"
            "explicit human action.\n\n"
            "Machine review remains advisory only. Settlement writes are human\n"
            "GitHub reviews/comments plus local founder-review receipts. See\n"
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
    packet_p.add_argument(
        "--execute-reviewers",
        action="store_true",
        help=(
            "Attempt one bounded live heterogeneous reviewer pass before falling back to "
            "the metadata-derived packet."
        ),
    )
    packet_p.add_argument("--json", action="store_true", help="Output as JSON")

    run_p = sub.add_parser("run", help="Interactively settle a prioritized PR queue")
    run_p.add_argument("--limit", type=int, default=30, help="Max PRs to walk (default: 30)")
    run_p.add_argument(
        "--ready-only",
        action="store_true",
        help="Restrict the session to ready_now items",
    )
    run_p.add_argument(
        "--include-parked",
        action="store_true",
        help="Include parked items in the session",
    )
    run_p.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override (owner/name). Defaults to current repo context.",
    )

    act_p = sub.add_parser("act", help="Settle one PR with a human action")
    act_p.add_argument("pr", help="PR number")
    act_p.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override (owner/name). Defaults to current repo context.",
    )
    act_group = act_p.add_mutually_exclusive_group(required=True)
    act_group.add_argument("--approve", action="store_true", help="Post a human APPROVE review")
    act_group.add_argument(
        "--request-changes",
        action="store_true",
        help="Post a human REQUEST_CHANGES review",
    )
    act_group.add_argument("--defer", action="store_true", help="Leave a human defer comment")
    act_p.add_argument(
        "--reason",
        default="",
        help="One-line human reason (required for --request-changes and --defer)",
    )
    act_p.add_argument("--json", action="store_true", help="Output settlement receipt as JSON")

    parser.set_defaults(func=cmd_review_queue)


def cmd_review_queue(args: argparse.Namespace) -> int:
    """Dispatch review-queue subcommands."""
    command = getattr(args, "review_queue_command", None)
    if command == "build":
        return _cmd_build(args)
    if command == "packet":
        return _cmd_packet(args)
    if command == "run":
        return _cmd_run(args)
    if command == "act":
        return _cmd_act(args)
    print(
        "Usage: aragora review-queue {build,packet,run,act} [...]\n"
        "Run 'aragora review-queue run --help' for the human settlement loop.",
        file=sys.stderr,
    )
    return 2


# --- Subcommand entry points -----------------------------------------------


def _cmd_build(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False) or getattr(args, "json_output", False))
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
    if json_output:
        print(json.dumps([item.to_dict() for item in items], indent=2))
    else:
        _render_table(items)
    return 0


def _cmd_packet(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False) or getattr(args, "json_output", False))
    try:
        packet = _build_packet(
            args.pr,
            repo_override=args.repo,
            execute_reviewers=bool(getattr(args, "execute_reviewers", False)),
        )
    except _GhError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if json_output:
        print(json.dumps(packet.to_dict(), indent=2))
    else:
        _render_packet(packet)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    repo_root = resolve_repo_root(Path.cwd())
    try:
        _require_clean_worktree(repo_root)
        items = _build_queue(limit=args.limit)
    except _GhError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    items = _filter_lanes(
        items,
        ready_only=bool(getattr(args, "ready_only", False)),
        include_parked=bool(getattr(args, "include_parked", False)),
    )
    if not items:
        print("(no PRs in scope)")
        return 0

    session_id = _session_id()
    started_at = datetime.now(UTC).isoformat()
    session_receipt: dict[str, Any] = {
        "session_id": session_id,
        "started_at": started_at,
        "completed_at": None,
        "reviewed_prs": [],
        "queue_size": len(items),
        "ready_only": bool(getattr(args, "ready_only", False)),
        "include_parked": bool(getattr(args, "include_parked", False)),
    }
    session_path = _session_receipt_path(repo_root, session_id)
    _write_json(session_path, session_receipt)

    for index, item in enumerate(items, start=1):
        try:
            packet = _build_packet(str(item.number), repo_override=getattr(args, "repo", None))
        except _GhError as exc:
            print(f"\n! skipped PR #{item.number}: {exc}", file=sys.stderr)
            continue
        _render_session_packet(packet, item=item, index=index, total=len(items))
        decision_started = time.monotonic()

        while True:
            choice = input(
                "[a]pprove [r]equest-changes [d]efer [o]pen-files [p]acket-json [q]uit: "
            )
            normalized = choice.strip().lower()
            if normalized == "o":
                _render_changed_files(packet.pr_number, repo_override=getattr(args, "repo", None))
                continue
            if normalized == "p":
                print(json.dumps(packet.to_dict(), indent=2))
                continue
            if normalized == "q":
                session_receipt["completed_at"] = datetime.now(UTC).isoformat()
                _write_json(session_path, session_receipt)
                print(f"Session saved: {session_path}")
                return 0
            if normalized not in {"a", "r", "d"}:
                print("Choose one of: a, r, d, o, p, q")
                continue

            action = {
                "a": "approve",
                "r": "request_changes",
                "d": "defer",
            }[normalized]
            reason = ""
            if action == "approve":
                reason = input("approve note (optional): ").strip()
            else:
                while not reason:
                    prompt = "reason (required): "
                    reason = input(prompt).strip()
            try:
                receipt = _settle_packet(
                    packet=packet,
                    action=action,
                    reason=reason,
                    repo_root=repo_root,
                    repo_override=getattr(args, "repo", None),
                    session_id=session_id,
                    elapsed_seconds=round(time.monotonic() - decision_started, 3),
                )
            except _GhError as exc:
                print(f"! could not settle PR #{packet.pr_number}: {exc}", file=sys.stderr)
                continue
            session_receipt["reviewed_prs"].append(receipt.to_dict())
            _write_json(session_path, session_receipt)
            print(
                f"{action} recorded for PR #{packet.pr_number} "
                f"(packet {packet.packet_sha}, head {packet.head_sha})"
            )
            break

    session_receipt["completed_at"] = datetime.now(UTC).isoformat()
    _write_json(session_path, session_receipt)
    print(f"Session complete: {session_path}")
    return 0


def _cmd_act(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False) or getattr(args, "json_output", False))
    action = _requested_action(args)
    reason = str(getattr(args, "reason", "") or "").strip()
    if action == "request_changes" and not reason:
        print(f"error: {REQUEST_CHANGES_REASON_REQUIRED}", file=sys.stderr)
        return 2
    if action == "defer" and not reason:
        print(f"error: {DEFER_REASON_REQUIRED}", file=sys.stderr)
        return 2

    repo_root = resolve_repo_root(Path.cwd())
    try:
        _require_clean_worktree(repo_root)
        packet = _build_packet(args.pr, repo_override=getattr(args, "repo", None))
        receipt = _settle_packet(
            packet=packet,
            action=action,
            reason=reason,
            repo_root=repo_root,
            repo_override=getattr(args, "repo", None),
            session_id=_session_id(),
        )
    except _GhError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if json_output:
        print(json.dumps(receipt.to_dict(), indent=2))
    else:
        _render_settlement_receipt(receipt)
    return 0


# --- Internals: gh shell, classification, packet building ------------------


class _GhError(RuntimeError):
    """Raised when a 'gh' invocation fails or returns malformed JSON."""


def _gh_text(args: list[str]) -> str:
    """Run a 'gh' command and return plain stdout."""
    proc = subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "no stderr"
        raise _GhError(f"gh {' '.join(args)} failed: {stderr}")
    return proc.stdout.strip()


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


def _build_packet(
    pr_ref: str,
    *,
    repo_override: str | None,
    execute_reviewers: bool = False,
) -> ReviewPacket:
    number = _parse_pr_number(pr_ref)
    fields = ",".join(
        [
            "number",
            "title",
            "url",
            "headRefOid",
            "baseRefOid",
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
            "body",
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
    queue_item = _classify_pr(pr)
    validation = _extract_validation_commands(str(pr.get("body", "") or ""))

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

    repo = _repo_from_url(str(pr.get("url", "")).strip())
    protocol_runner = default_pr_review_protocol()
    reviewer_outputs: list[ReviewerOutput] = []
    execution_failures: list[PRReviewerExecutionFailure] = []
    if execute_reviewers:
        diff_args = ["pr", "diff", str(number)]
        if repo_override:
            diff_args.extend(["--repo", repo_override])
        reviewer_outputs, execution_failures = protocol_runner.execute_live_reviewers(
            repo=repo,
            pr_number=number,
            title=str(pr.get("title", "")).strip(),
            base_sha=str(pr.get("baseRefOid", "")).strip(),
            head_sha=str(pr.get("headRefOid", "")).strip(),
            checks_summary=checks_summary,
            changed_files=files,
            diff_text=_gh_text(diff_args),
            machine_recommendation=recommendation,
            machine_recommendation_reason=recommendation_reason,
        )
    protocol = protocol_runner.build_packet(
        repo=repo,
        pr_number=number,
        title=str(pr.get("title", "")).strip(),
        base_sha=str(pr.get("baseRefOid", "")).strip(),
        head_sha=str(pr.get("headRefOid", "")).strip(),
        mergeable=mergeable,
        review_decision=str(pr.get("reviewDecision", "")).strip().upper(),
        checks_summary=checks_summary,
        has_failures=has_failures,
        has_pending=has_pending,
        additions=additions,
        deletions=deletions,
        changed_files=int(pr.get("changedFiles", 0) or 0),
        labels=labels,
        high_risk_paths=high_risk,
        validation_commands=validation,
        machine_recommendation=recommendation,
        machine_recommendation_reason=recommendation_reason,
        reviewer_outputs=reviewer_outputs,
        execution_failures=execution_failures,
    )

    packet = ReviewPacket(
        pr_number=number,
        title=str(pr.get("title", "")).strip(),
        url=str(pr.get("url", "")).strip(),
        head_sha=str(pr.get("headRefOid", "")).strip(),
        base_sha=str(pr.get("baseRefOid", "")).strip(),
        author=author,
        is_draft=is_draft,
        additions=additions,
        deletions=deletions,
        changed_files=int(pr.get("changedFiles", 0) or 0),
        queue_bucket=queue_item.lane,
        touched_subsystems=touched,
        high_risk_paths_touched=high_risk,
        validation=validation,
        checks_summary=checks_summary,
        risk_flags=risk_flags,
        machine_recommendation=recommendation,
        machine_recommendation_reason=recommendation_reason,
        packet_sha="",
        generated_at=datetime.now(UTC).isoformat(),
        protocol=protocol.to_dict(),
    )
    packet.packet_sha = _packet_sha(packet)
    return packet


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


def _repo_from_url(url: str) -> str:
    text = str(url).strip().rstrip("/")
    parts = text.split("/")
    if len(parts) >= 5 and parts[2] == "github.com":
        return f"{parts[3]}/{parts[4]}"
    return ""


def _extract_validation_commands(body: str) -> list[str]:
    """Parse bullet lines from a conventional PR `## Validation` section."""
    lines: list[str] = []
    in_validation = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("## "):
            in_validation = lower == "## validation"
            continue
        if not in_validation:
            continue
        if line.startswith("### "):
            break
        if line.startswith("- ") or line.startswith("* "):
            lines.append(line[2:].strip())
    return lines


def _requested_action(args: argparse.Namespace) -> str:
    if bool(getattr(args, "approve", False)):
        return "approve"
    if bool(getattr(args, "request_changes", False)):
        return "request_changes"
    if bool(getattr(args, "defer", False)):
        return "defer"
    raise _GhError("no settlement action selected")


def _packet_sha(packet: ReviewPacket) -> str:
    payload = packet.to_dict()
    payload.pop("packet_sha", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _session_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _review_queue_root(repo_root: Path) -> Path:
    return repo_root / REVIEW_QUEUE_ARTIFACT_DIR


def _session_receipt_path(repo_root: Path, session_id: str) -> Path:
    return _review_queue_root(repo_root) / "sessions" / f"{session_id}.json"


def _settlement_receipt_path(
    repo_root: Path,
    *,
    session_id: str,
    pr_number: int,
    action: str,
) -> Path:
    filename = f"pr-{pr_number}-{session_id}-{action}.json"
    return _review_queue_root(repo_root) / "receipts" / filename


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _require_clean_worktree(repo_root: Path) -> None:
    proc = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or "no stderr"
        raise _GhError(f"git status failed in {repo_root}: {stderr}")
    if proc.stdout.strip():
        raise _GhError(
            "review-queue settlement requires a clean worktree so receipts match the reviewed truth"
        )


def _current_head_sha(pr_number: int, *, repo_override: str | None) -> str:
    args = ["pr", "view", str(pr_number), "--json", "headRefOid"]
    if repo_override:
        args.extend(["--repo", repo_override])
    payload = _gh_json(args)
    if not isinstance(payload, dict):
        raise _GhError(f"PR #{pr_number} not found while verifying packet freshness")
    return str(payload.get("headRefOid", "")).strip()


def _github_actor() -> str:
    try:
        payload = _gh_json(["api", "user"])
    except _GhError:
        return "unknown"
    if not isinstance(payload, dict):
        return "unknown"
    return str(payload.get("login", "unknown") or "unknown").strip()


def _github_settlement_event(action: str) -> str:
    if action == "approve":
        return "APPROVE"
    if action == "request_changes":
        return "REQUEST_CHANGES"
    return "COMMENT"


def _settlement_body(packet: ReviewPacket, *, action: str, reason: str) -> str:
    lines = [
        "Human settlement via `aragora review-queue`.",
        "",
        f"- Action: `{action}`",
        f"- Packet SHA: `{packet.packet_sha}`",
        f"- Head SHA: `{packet.head_sha}`",
        f"- Base SHA: `{packet.base_sha}`",
        f"- Queue bucket: `{packet.queue_bucket}`",
        f"- Machine recommendation: `{packet.machine_recommendation}`",
    ]
    if reason:
        lines.append(f"- Reason: {reason}")
    lines.extend(
        [
            "",
            ADVISORY_NOTE,
        ]
    )
    return "\n".join(lines)


def _settle_packet(
    *,
    packet: ReviewPacket,
    action: str,
    reason: str,
    repo_root: Path,
    repo_override: str | None,
    session_id: str,
    elapsed_seconds: float | None = None,
) -> SettlementReceipt:
    current_head_sha = _current_head_sha(packet.pr_number, repo_override=repo_override)
    if current_head_sha != packet.head_sha:
        raise _GhError(
            f"PR #{packet.pr_number} head changed from {packet.head_sha} to {current_head_sha}; "
            "refresh the packet before settlement"
        )

    body = _settlement_body(packet, action=action, reason=reason)
    if action == "approve":
        gh_args = ["pr", "review", str(packet.pr_number), "--approve", "--body", body]
    elif action == "request_changes":
        gh_args = ["pr", "review", str(packet.pr_number), "--request-changes", "--body", body]
    else:
        gh_args = ["pr", "comment", str(packet.pr_number), "--body", body]
    if repo_override:
        gh_args.extend(["--repo", repo_override])
    _gh_text(gh_args)

    reviewed_at = datetime.now(UTC).isoformat()
    receipt = SettlementReceipt(
        session_id=session_id,
        reviewed_at=reviewed_at,
        actor=_github_actor(),
        action=action,
        reason=reason,
        pr_number=packet.pr_number,
        pr_url=packet.url,
        head_sha=packet.head_sha,
        base_sha=packet.base_sha,
        packet_sha=packet.packet_sha,
        queue_bucket=packet.queue_bucket,
        machine_recommendation=packet.machine_recommendation,
        github_event=_github_settlement_event(action),
        elapsed_seconds=elapsed_seconds,
    )
    receipt_path = _settlement_receipt_path(
        repo_root,
        session_id=session_id,
        pr_number=packet.pr_number,
        action=action,
    )
    receipt.receipt_path = str(receipt_path)
    _write_json(receipt_path, receipt.to_dict())
    return receipt


def _render_changed_files(pr_number: int, *, repo_override: str | None) -> None:
    args = ["pr", "view", str(pr_number), "--json", "files"]
    if repo_override:
        args.extend(["--repo", repo_override])
    payload = _gh_json(args)
    files = []
    if isinstance(payload, dict):
        for item in payload.get("files") or []:
            if isinstance(item, dict) and item.get("path"):
                files.append(str(item["path"]).strip())
    print("changed files:")
    for path in files:
        print(f"  - {path}")
    if not files:
        print("  (no changed files reported)")


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
        "Note: machine review remains advisory only. Settlement writes are explicit "
        "human `gh pr review` / `gh pr comment` actions with local receipts."
    )


def _render_packet(packet: ReviewPacket) -> None:
    print(f"# Advisory review packet — PR #{packet.pr_number}")
    print(f"# {packet.title}")
    print(f"# {packet.url}")
    print()
    print(f"head SHA:        {packet.head_sha}")
    print(f"base SHA:        {packet.base_sha}")
    print(f"packet SHA:      {packet.packet_sha}")
    print(f"author:          {packet.author}")
    print(f"draft:           {packet.is_draft}")
    print(f"queue bucket:    {packet.queue_bucket}")
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
    if packet.validation:
        print("validation:")
        for line in packet.validation:
            print(f"  - {line}")
        print()
    if packet.risk_flags:
        print("risk flags:")
        for flag in packet.risk_flags:
            print(f"  - {flag}")
        print()
    print(f"machine recommendation: {packet.machine_recommendation}")
    print(f"  reason: {packet.machine_recommendation_reason}")
    if packet.protocol:
        protocol = packet.protocol
        binding = protocol.get("binding") or {}
        cost_estimate = protocol.get("cost_estimate") or {}
        print()
        print("protocol:")
        print(
            f"  {protocol.get('protocol_version', 'unknown')} [{protocol.get('status', 'unknown')}]"
        )
        print(
            f"  binding: {binding.get('repo', '')} "
            f"PR #{binding.get('pr_number', packet.pr_number)} "
            f"{binding.get('base_sha', packet.base_sha)}..{binding.get('head_sha', packet.head_sha)}"
        )
        print(
            f"  confidence: {protocol.get('confidence', 0):.2f} "
            f"({protocol.get('confidence_basis', 'unknown')})"
        )
        print(f"  dissent: {protocol.get('dissent_summary', '')}")
        availability_summary = protocol.get("availability_summary") or {}
        if availability_summary:
            print(
                "  availability: "
                f"{availability_summary.get('resolved_slots', 0)}/"
                f"{availability_summary.get('total_slots', 0)} slots resolved"
            )
            unresolved_slots = availability_summary.get("unresolved_slots") or []
            if unresolved_slots:
                unresolved = ", ".join(str(slot) for slot in unresolved_slots)
                print(f"    unresolved: {unresolved}")
            opt_in_slots = availability_summary.get("opt_in_slots") or []
            if opt_in_slots:
                opt_in = ", ".join(str(slot) for slot in opt_in_slots)
                print(f"    opt-in: {opt_in}")
        print(
            f"  cost estimate: ${cost_estimate.get('low', 0):.2f}"
            f"-${cost_estimate.get('high', 0):.2f}"
        )
        top_findings = protocol.get("top_findings") or []
        if top_findings:
            print("  top findings:")
            for finding in top_findings[:3]:
                if not isinstance(finding, dict):
                    continue
                severity = str(finding.get("severity", "")).strip()
                summary = str(finding.get("summary", "")).strip()
                print(f"    - [{severity}] {summary}")
        provider_slots = protocol.get("provider_slots") or []
        if provider_slots:
            print("  provider slots:")
            for slot in provider_slots:
                if not isinstance(slot, dict):
                    continue
                selected = slot.get("selected_provider") or "unresolved"
                print(
                    f"    - {slot.get('slot_id')}: {selected} "
                    f"({slot.get('family')}/{slot.get('lens')})"
                )
    print()
    print(f"generated at: {packet.generated_at}")
    print()
    print(f"-- {packet.settlement_note}")


def _render_session_packet(
    packet: ReviewPacket,
    *,
    item: QueueItem,
    index: int,
    total: int,
) -> None:
    print()
    print(f"[{index}/{total}] lane={item.lane}  reason={item.lane_reason}")
    _render_packet(packet)


def _render_settlement_receipt(receipt: SettlementReceipt) -> None:
    print(f"Recorded {receipt.action} for PR #{receipt.pr_number}")
    print(f"  actor:        {receipt.actor}")
    print(f"  reviewed at:  {receipt.reviewed_at}")
    print(f"  head SHA:     {receipt.head_sha}")
    print(f"  packet SHA:   {receipt.packet_sha}")
    print(f"  queue bucket: {receipt.queue_bucket}")
    print(f"  event:        {receipt.github_event}")
    if receipt.reason:
        print(f"  reason:       {receipt.reason}")
    if receipt.elapsed_seconds is not None:
        print(f"  elapsed:      {receipt.elapsed_seconds:.3f}s")
    print(f"  receipt:      {receipt.receipt_path}")
