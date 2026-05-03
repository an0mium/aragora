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
import sqlite3
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.review.invalidation import (
    BaselineMeasurement,
    DEFAULT_BASELINE_WINDOW_DAYS,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_MINIMUM_MEANINGFUL_RATE,
    DEFAULT_SAFETY_MARGIN,
    ThresholdProposal,
    derive_threshold,
)
from aragora.review.invalidation_event_source import measure_baseline_from_stores
from aragora.review.reviewer_output import ReviewerOutput
from aragora.swarm.pr_review_protocol import (
    PRReviewerExecutionFailure,
    default_pr_review_protocol,
)
from aragora.triage.auto_handle_calibration import AutoHandleCalibrationStore
from aragora.worktree.fleet import resolve_repo_root

UTC = timezone.utc

# Lane classification thresholds and risk-path catalog.
LARGE_DIFF_THRESHOLD = 500  # additions + deletions, beyond which "needs_human_attention"
MODEL_REVIEW_QUEUE_CAP = 6
MODEL_REVIEW_QUORUM_VERSION = "model_review_quorum.v1"
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
TIER_2_PREFIXES: tuple[str, ...] = (
    "aragora/cli/",
    "aragora/swarm/",
    "aragora/observability/",
    "aragora/knowledge/mound/metrics",
    "scripts/",
)
TIER_3_PREFIXES: tuple[str, ...] = (
    "aragora/auth/",
    "aragora/rbac/",
    "aragora/security/",
    "aragora/privacy/",
    "aragora/compliance/",
    "aragora/metrics/",
    "aragora/reputation/",
    "aragora/debate/team_selector.py",
    "aragora/server/fastapi/routes/",
    "aragora/server/handlers/",
    "aragora/migrations/",
    "sdk/",
)
TIER_3_TITLE_KEYWORDS: tuple[str, ...] = (
    "agt-",
    "calibration",
    "reputation",
    "semantic",
    "scoring",
    "persistence",
    "public api",
)
TIER_4_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "deploy/",
    "docker/",
    "k8s/",
    # Merge-authority self-modification: when a PR changes the code that
    # enforces model-quorum settlement gates, that PR's own quorum is
    # evaluated by the version of the gate it is trying to land. A bug or
    # weakening introduced in the diff would let the diff itself through.
    # Elevate to Tier 4 (human preapproval) so the human chain-of-trust is
    # not delegated to the artifact under review.
    "aragora/cli/commands/review_queue.py",
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
    model_review_quorum: dict[str, Any] = field(default_factory=dict)
    advisory_only: bool = True
    settlement_note: str = ADVISORY_NOTE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SettlementReceipt:
    """Persisted human settlement receipt for one PR/head/packet tuple.

    The five ``outcome_*`` fields are optional post-settlement signals that
    correspond exactly to the canonical invalidation labels in
    :data:`aragora.review.invalidation.INVALIDATION_SIGNALS`. They default to
    ``None`` (= "signal not yet observed") to preserve backward compatibility
    with receipts written before #6375 phase 4. When ``observe_outcome`` (in
    :mod:`aragora.review.settlement_outcome`) populates these, downstream
    consumers like :mod:`aragora.review.invalidation_event_source` can finally
    classify the human side of the baseline (closing the
    ``schema_gap_human_numerator`` note that #6898 surfaces).

    Semantics:
      - All five ``None`` → receipt is denominator-only (counted in
        ``total_human_settled`` but not in invalidation numerator).
      - Any one ``True`` → receipt is invalidated, all firing signals
        contribute to the numerator.
      - All five ``False`` → receipt is a clean human-settled non-invalidation
        (still denominator, explicitly not numerator).

    ``outcome_observed_at`` is the ISO 8601 UTC timestamp at which the
    observation was recorded (separate from ``reviewed_at`` which is the
    settlement time). ``None`` iff none of the five outcome fields have been
    observed yet.
    """

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
    # Post-settlement outcome signals (#6375 phase 4). None == not yet observed.
    outcome_revert_within_window: bool | None = None
    outcome_post_merge_incident: bool | None = None
    outcome_human_override_redo: bool | None = None
    outcome_rollback: bool | None = None
    outcome_reopened_pr: bool | None = None
    outcome_observed_at: str | None = None

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

    merge_packet_p = sub.add_parser(
        "merge-packet",
        help="Print a model-quorum merge authorization packet for a PR batch",
        description=(
            "Build a receipt-shaped batch packet for the Model Review Quorum + "
            "Human Risk Settlement process. This is read-only: it does not "
            "approve, merge, comment, or write local receipts."
        ),
    )
    merge_packet_p.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max open PRs to inspect when --pr is not supplied (default: 30)",
    )
    merge_packet_p.add_argument(
        "--pr",
        action="append",
        default=[],
        help="Specific PR number/ref to include. Repeatable. Defaults to open queue.",
    )
    merge_packet_p.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override (owner/name). Defaults to current repo context.",
    )
    merge_packet_p.add_argument(
        "--execute-reviewers",
        action="store_true",
        help="Attempt live heterogeneous reviewer execution for each packet.",
    )
    merge_packet_p.add_argument("--json", action="store_true", help="Output as JSON")

    baseline_p = sub.add_parser(
        "baseline",
        help="Measure empirical invalidation baseline from on-disk stores (#6375)",
        description=(
            "Read the auto-handle calibration store (failure outcomes) and the\n"
            "settlement-receipt tree (denominator for human-settled decisions),\n"
            "compute the empirical invalidation baseline, and propose an auto-\n"
            "handle invalidation threshold.\n\n"
            "Read-only: this command does NOT mutate the calibration store, the\n"
            "receipt tree, or any threshold configuration. It is the operator-\n"
            "facing surface for the empirical-threshold framework that landed\n"
            "in #6602 (phase 1) and #6615 (phase 2 adapter)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    baseline_p.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_BASELINE_WINDOW_DAYS,
        help=(f"Measurement-window width in days (default: {DEFAULT_BASELINE_WINDOW_DAYS})."),
    )
    baseline_p.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_BASELINE_SAMPLES,
        help=(
            "Minimum human-settled sample size before the baseline is "
            f"considered usable for non-placeholder threshold derivation "
            f"(default: {DEFAULT_MIN_BASELINE_SAMPLES})."
        ),
    )
    baseline_p.add_argument(
        "--safety-margin",
        type=float,
        default=DEFAULT_SAFETY_MARGIN,
        help=(
            "Multiplier applied to the baseline when deriving the threshold "
            f"(default: {DEFAULT_SAFETY_MARGIN}). Must be in (0, 1]."
        ),
    )
    baseline_p.add_argument(
        "--minimum-meaningful-rate",
        type=float,
        default=DEFAULT_MINIMUM_MEANINGFUL_RATE,
        help=(
            "Floor below which threshold drift is indistinguishable from "
            f"sample noise (default: {DEFAULT_MINIMUM_MEANINGFUL_RATE})."
        ),
    )
    baseline_p.add_argument(
        "--placeholder-value",
        type=float,
        default=0.05,
        help=(
            "Threshold to use when the baseline is below the sample-size "
            "floor (default: 0.05, matching the THESIS Commitment 3 placeholder)."
        ),
    )
    baseline_p.add_argument(
        "--calibration-db",
        default=None,
        help=(
            "Override the auto-handle calibration store path. Defaults to "
            "the canonical store under aragora's data dir."
        ),
    )
    baseline_p.add_argument(
        "--review-queue-root",
        default=None,
        help=(
            "Override the review-queue store root used for settlement "
            "receipts. Defaults to <repo>/.aragora/review-queue."
        ),
    )
    baseline_p.add_argument(
        "--json",
        action="store_true",
        help="Output the BaselineMeasurement + ThresholdProposal as JSON.",
    )

    from aragora.review.observe_outcomes_cli import add_observe_outcomes_subparser

    add_observe_outcomes_subparser(sub)

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
    if command == "merge-packet":
        return _cmd_merge_packet(args)
    if command == "baseline":
        return _cmd_baseline(args)
    if command == "observe-outcomes":
        from aragora.cli.commands.observe_outcomes_cmd import cmd_observe_outcomes

        return cmd_observe_outcomes(args)
    print(
        "Usage: aragora review-queue "
        "{build,packet,run,act,merge-packet,baseline,observe-outcomes} [...]\n"
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


def _cmd_merge_packet(args: argparse.Namespace) -> int:
    json_output = bool(getattr(args, "json", False) or getattr(args, "json_output", False))
    try:
        packet = _build_merge_authorization_packet(
            pr_refs=list(getattr(args, "pr", []) or []),
            limit=int(getattr(args, "limit", 30) or 30),
            repo_override=getattr(args, "repo", None),
            execute_reviewers=bool(getattr(args, "execute_reviewers", False)),
        )
    except _GhError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if json_output:
        print(json.dumps(packet, indent=2))
    else:
        _render_merge_authorization_packet(packet)
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Measure the empirical invalidation baseline + propose a threshold.

    Read-only against the calibration store and the settlement-receipt
    tree. Does not mutate either; does not write a receipt; does not
    apply the proposed threshold anywhere. The recalibration receipt
    flow is #6375 step B (codex).
    """
    if args.window_days <= 0:
        print("error: --window-days must be positive", file=sys.stderr)
        return 2
    if args.min_samples <= 0:
        print("error: --min-samples must be positive", file=sys.stderr)
        return 2
    if not 0 < args.safety_margin <= 1:
        print("error: --safety-margin must be in (0, 1]", file=sys.stderr)
        return 2
    if args.minimum_meaningful_rate <= 0:
        print("error: --minimum-meaningful-rate must be positive", file=sys.stderr)
        return 2
    if not 0 < args.placeholder_value < 1:
        print("error: --placeholder-value must be in (0, 1)", file=sys.stderr)
        return 2

    json_output = bool(getattr(args, "json", False))

    try:
        store = AutoHandleCalibrationStore(db_path=args.calibration_db)
    except (OSError, RuntimeError, sqlite3.Error, ValueError, TypeError) as exc:
        print(f"error: cannot open calibration store: {exc}", file=sys.stderr)
        return 1

    window_end = datetime.now(UTC)
    try:
        measurement = measure_baseline_from_stores(
            calibration_store=store,
            review_queue_root=args.review_queue_root,
            window_end=window_end,
            window_days=args.window_days,
            min_samples=args.min_samples,
        )
    except (OSError, RuntimeError, sqlite3.Error, ValueError) as exc:
        print(f"error: baseline measurement failed: {exc}", file=sys.stderr)
        return 1

    proposal = derive_threshold(
        measurement,
        safety_margin=args.safety_margin,
        minimum_meaningful_rate=args.minimum_meaningful_rate,
        measured_at=window_end,
        placeholder_value=args.placeholder_value,
    )

    if json_output:
        print(
            json.dumps(
                {
                    "measurement": measurement.to_dict(),
                    "proposal": proposal.to_dict(),
                },
                indent=2,
            )
        )
    else:
        _render_baseline_report(measurement=measurement, proposal=proposal)
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
            "comments",
            "commits",
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
    protocol_dict = protocol.to_dict()

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
        protocol=protocol_dict,
        model_review_quorum=_build_model_review_quorum(
            pr=pr,
            files=files,
            protocol=protocol_dict,
            machine_recommendation=recommendation,
            has_pending=has_pending,
            has_failures=has_failures,
        ),
    )
    packet.packet_sha = _packet_sha(packet)
    return packet


def _build_merge_authorization_packet(
    *,
    pr_refs: list[str],
    limit: int,
    repo_override: str | None,
    execute_reviewers: bool = False,
) -> dict[str, Any]:
    if pr_refs:
        refs = list(dict.fromkeys(str(ref).strip() for ref in pr_refs if str(ref).strip()))
        queue_size = len(_build_queue(limit=max(limit, len(refs), MODEL_REVIEW_QUEUE_CAP + 1)))
    else:
        queue = _build_queue(limit=limit)
        refs = [str(item.number) for item in queue]
        queue_size = len(queue)

    packets = [
        _build_packet(ref, repo_override=repo_override, execute_reviewers=execute_reviewers)
        for ref in refs
    ]
    queue_pressure_active = queue_size > MODEL_REVIEW_QUEUE_CAP
    entries = []
    for packet in packets:
        quorum = dict(packet.model_review_quorum)
        quorum["queue_pressure"] = {
            "current_open_prs": queue_size,
            "cap": MODEL_REVIEW_QUEUE_CAP,
            "active": queue_pressure_active,
            "allowed_work_when_active": [
                "review",
                "dogfood",
                "existing_blocker_fix",
                "local_spec_only",
                "merge_authorization_packet",
            ],
        }
        entry = {
            "pr_number": packet.pr_number,
            "title": packet.title,
            "url": packet.url,
            "head_sha": packet.head_sha,
            "checks_summary": packet.checks_summary,
            "machine_recommendation": packet.machine_recommendation,
            "tier": quorum["tier"],
            "tier_name": quorum["tier_name"],
            "status": quorum["status"],
            "verdict": quorum["verdict"],
            "admin_squash_allowed": quorum["admin_squash_allowed"],
            "requires_human_risk_settlement": quorum["requires_human_risk_settlement"],
            "unresolved_dissent": quorum["unresolved_dissent"],
            "reviewer_signals": quorum["reviewer_signals"],
            "dogfood_evidence": quorum["dogfood_evidence"],
            "counted_reviewer_ids": quorum["counted_reviewer_ids"],
            "reasons": quorum["reasons"],
        }
        entries.append(entry)

    return {
        "version": "merge_authorization_packet.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "queue_pressure": {
            "current_open_prs": queue_size,
            "cap": MODEL_REVIEW_QUEUE_CAP,
            "active": queue_pressure_active,
        },
        "authorization_sentence": (
            "I accept the model quorum evidence for Tier 0-2 PRs in this packet "
            "and authorize admin squash in the listed order. For Tier 3+ PRs, "
            "I separately accept the semantic-risk packet before merge."
        ),
        "entries": entries,
        "admin_squash_order": [
            entry["pr_number"]
            for entry in entries
            if bool(entry["admin_squash_allowed"]) and not bool(entry["unresolved_dissent"])
        ],
        "human_risk_settlement_required": [
            entry["pr_number"] for entry in entries if bool(entry["requires_human_risk_settlement"])
        ],
        "not_ready": [
            entry["pr_number"]
            for entry in entries
            if entry["status"] not in {"satisfied", "human_risk_settlement_required"}
        ],
    }


def _build_model_review_quorum(
    *,
    pr: dict[str, Any],
    files: list[str],
    protocol: dict[str, Any],
    machine_recommendation: str,
    has_pending: bool,
    has_failures: bool,
) -> dict[str, Any]:
    tier, tier_name, tier_reason = _classify_model_review_tier(files, pr=pr)
    requirement = _tier_requirement(tier)
    head_sha = str(pr.get("headRefOid", "") or "").strip()
    head_committed_at = _head_committed_at_from_pr(pr)
    reviewer_signals = _reviewer_signals_from_protocol(protocol)
    reviewer_signals.extend(
        _model_review_signals_from_comments(
            pr.get("comments") or [],
            head_sha=head_sha,
            head_committed_at=head_committed_at,
        )
    )
    dogfood_evidence = _dogfood_evidence_from_comments(
        pr.get("comments") or [],
        head_sha=head_sha,
        head_committed_at=head_committed_at,
    )
    dissenting_views = [
        view for view in (protocol.get("dissenting_views") or []) if isinstance(view, dict)
    ]
    blocking_workflow_state = _has_blocking_workflow_state(pr)
    unresolved_dissent = bool(dissenting_views)
    counted_reviewer_ids = _counted_model_reviewer_ids(reviewer_signals, dogfood_evidence)
    signal_count = len(counted_reviewer_ids)
    has_required_dogfood = not requirement["requires_adversarial_dogfood"] or any(
        _known_model_reviewer_id(item) for item in dogfood_evidence
    )
    quorum_satisfied = (
        signal_count >= requirement["required_model_signals"] and has_required_dogfood
    )

    reasons = [tier_reason]
    if has_failures:
        reasons.append("checks are failing; repair before settlement")
    if has_pending:
        reasons.append("checks are pending; wait before settlement")
    if unresolved_dissent:
        reasons.append("unresolved model dissent is present")
    if not quorum_satisfied:
        reasons.append(
            "model quorum incomplete: "
            f"{signal_count}/{requirement['required_model_signals']} signal(s)"
        )
        if not has_required_dogfood:
            reasons.append("focused adversarial dogfood evidence is required")

    admin_squash_allowed = False
    requires_human_risk_settlement = bool(requirement["requires_human_risk_settlement"])
    if (
        has_failures
        or has_pending
        or machine_recommendation == "repair_first"
        or blocking_workflow_state
    ):
        status = "repair_or_wait"
        verdict = "not_ready_for_settlement"
    elif not quorum_satisfied:
        status = "needs_model_review_quorum"
        verdict = "collect_model_quorum_before_merge"
    elif unresolved_dissent:
        status = "unresolved_dissent"
        verdict = "human_risk_settlement_required"
        requires_human_risk_settlement = True
    elif requirement["requires_human_preapproval"]:
        status = "human_preapproval_required"
        verdict = "tier_4_human_preapproval_required"
        requires_human_risk_settlement = True
    elif requires_human_risk_settlement:
        status = "human_risk_settlement_required"
        verdict = "model_quorum_satisfied_human_risk_settlement_required"
    else:
        status = "satisfied"
        verdict = "admin_squash_allowed"
        admin_squash_allowed = True

    return {
        "version": MODEL_REVIEW_QUORUM_VERSION,
        "head_sha": str(pr.get("headRefOid", "")).strip(),
        "tier": tier,
        "tier_name": tier_name,
        "tier_reason": tier_reason,
        "required_model_signals": requirement["required_model_signals"],
        "requires_adversarial_dogfood": requirement["requires_adversarial_dogfood"],
        "requires_human_risk_settlement": requires_human_risk_settlement,
        "requires_human_preapproval": requirement["requires_human_preapproval"],
        "admin_squash_allowed": admin_squash_allowed,
        "status": status,
        "verdict": verdict,
        "reviewer_signals": reviewer_signals,
        "dogfood_evidence": dogfood_evidence,
        "counted_reviewer_ids": counted_reviewer_ids,
        "dissenting_views": dissenting_views,
        "unresolved_dissent": unresolved_dissent,
        "reasons": reasons,
    }


def _classify_model_review_tier(
    files: list[str],
    *,
    pr: dict[str, Any] | None = None,
) -> tuple[int, str, str]:
    normalized = [path.strip() for path in files if path.strip()]
    title = str((pr or {}).get("title", "") or "").lower()
    if not normalized:
        return (1, "tier_1_additive_internal", "no changed files reported; defaulting to Tier 1")
    if any(_matches_prefix(path, TIER_4_PREFIXES) for path in normalized):
        return (4, "tier_4_preapproval_required", "workflow/deploy/destructive surface touched")
    if any(_matches_prefix(path, TIER_3_PREFIXES) for path in normalized) or any(
        keyword in title for keyword in TIER_3_TITLE_KEYWORDS
    ):
        return (
            3,
            "tier_3_semantic_risk",
            "semantic, persistence, security, API, or SDK surface touched",
        )
    if all(_is_docs_tests_or_status_path(path) for path in normalized):
        return (0, "tier_0_docs_tests_status", "docs/tests/status-only change")
    if any(_matches_prefix(path, TIER_2_PREFIXES) for path in normalized) or any(
        word in title for word in ("retry", "cache", "cli", "automation", "observability")
    ):
        return (
            2,
            "tier_2_live_automation",
            "live automation, CLI, observability, retry, or cache surface touched",
        )
    return (1, "tier_1_additive_internal", "bounded internal code surface")


def _tier_requirement(tier: int) -> dict[str, Any]:
    if tier <= 0:
        return {
            "required_model_signals": 1,
            "requires_adversarial_dogfood": False,
            "requires_human_risk_settlement": False,
            "requires_human_preapproval": False,
        }
    if tier == 1:
        return {
            "required_model_signals": 2,
            "requires_adversarial_dogfood": True,
            "requires_human_risk_settlement": False,
            "requires_human_preapproval": False,
        }
    if tier == 2:
        return {
            "required_model_signals": 2,
            "requires_adversarial_dogfood": True,
            "requires_human_risk_settlement": False,
            "requires_human_preapproval": False,
        }
    if tier == 3:
        return {
            "required_model_signals": 2,
            "requires_adversarial_dogfood": True,
            "requires_human_risk_settlement": True,
            "requires_human_preapproval": False,
        }
    return {
        "required_model_signals": 2,
        "requires_adversarial_dogfood": True,
        "requires_human_risk_settlement": True,
        "requires_human_preapproval": True,
    }


def _has_blocking_workflow_state(pr: dict[str, Any]) -> bool:
    if bool(pr.get("isDraft", False)):
        return True
    mergeable = str(pr.get("mergeable", "")).strip().upper()
    if mergeable == "CONFLICTING":
        return True
    labels = [
        str(label.get("name", "")).strip()
        for label in (pr.get("labels") or [])
        if isinstance(label, dict) and label.get("name")
    ]
    return any(label in PARKED_LABELS for label in labels)


def _reviewer_signals_from_protocol(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    validation = protocol.get("validation_summary") or {}
    reviewer_execution = validation.get("reviewer_execution") or {}
    reviewer_ids = reviewer_execution.get("reviewer_ids") or []
    providers = reviewer_execution.get("providers") or []
    signals = []
    for index, reviewer_id in enumerate(reviewer_ids):
        provider = providers[index] if index < len(providers) else ""
        signals.append(
            {
                "reviewer_id": str(reviewer_id),
                "provider": str(provider),
                "source": protocol.get("status", ""),
            }
        )
    return signals


def _counted_model_reviewer_ids(
    reviewer_signals: list[dict[str, Any]],
    dogfood_evidence: list[dict[str, str]],
) -> list[str]:
    reviewer_ids: set[str] = set()
    for item in [*reviewer_signals, *dogfood_evidence]:
        reviewer_id = _known_model_reviewer_id(item)
        if reviewer_id:
            reviewer_ids.add(reviewer_id)
    return sorted(reviewer_ids)


def _head_committed_at_from_pr(pr: dict[str, Any]) -> str:
    """Return the ``committedDate`` of the PR head commit, or ``""``.

    Used to anchor comment-based quorum signals to the current head
    SHA per the "grounded in the current head SHA" requirement of
    ``docs/REVIEW_AUTHORITY_PRINCIPLES.md``.  Falls back to the most
    recent ``committedDate`` in the commits list when the head SHA
    is not separately matched, and returns ``""`` when the PR fetch
    did not include commit metadata (no-op for legacy callers).
    """
    head_sha = str(pr.get("headRefOid", "") or "").strip()
    commits = pr.get("commits") or []
    if not isinstance(commits, list):
        return ""
    latest_committed_at = ""
    for entry in commits:
        if not isinstance(entry, dict):
            continue
        committed_at = str(entry.get("committedDate", "") or "").strip()
        if not committed_at:
            continue
        oid = str(entry.get("oid", "") or "").strip()
        if head_sha and oid == head_sha:
            return committed_at
        if committed_at > latest_committed_at:
            latest_committed_at = committed_at
    return latest_committed_at


def _known_model_reviewer_id(item: dict[str, Any]) -> str:
    provider = str(item.get("provider", "") or "")
    reviewer_id = str(item.get("reviewer_id", "") or "")
    return _normalize_model_reviewer_id(provider) or _normalize_model_reviewer_id(reviewer_id)


def _normalize_model_reviewer_id(value: str) -> str:
    lower = str(value).lower()
    if not lower or "unknown_model_reviewer" in lower:
        return ""
    known_markers = (
        ("claude", ("claude", "anthropic")),
        ("codex", ("codex",)),
        ("openai", ("openai", "gpt")),
        ("grok", ("grok", "xai")),
        ("gemini", ("gemini", "google")),
        ("mistral", ("mistral", "codestral")),
        ("deepseek", ("deepseek",)),
        ("qwen", ("qwen",)),
        ("kimi", ("kimi", "moonshot")),
        ("tesla", ("tesla",)),
        ("harvey", ("harvey",)),
        ("factory", ("factory",)),
    )
    for normalized, markers in known_markers:
        if any(marker in lower for marker in markers):
            return normalized
    return ""


def _dogfood_evidence_from_comments(
    comments: list[Any],
    *,
    head_sha: str = "",
    head_committed_at: str = "",
) -> list[dict[str, str]]:
    """Extract focused-adversarial dogfood signals from PR comments.

    Mirrors the source-side filtering of
    :func:`_model_review_signals_from_comments` for symmetry: an entry is
    only emitted when (a) the comment is SHA-grounded on the current head,
    (b) a known model reviewer can be inferred from the comment's
    structured header, and (c) the comment was not posted by GitHub
    Actions. Unknowns are still neutralised at counting time by
    :func:`_known_model_reviewer_id`, but excluding them at the source
    keeps the evidence list interpretable for downstream consumers.
    """
    evidence: list[dict[str, str]] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        if not _is_comment_grounded_on_head(comment, head_sha, head_committed_at):
            continue
        body = str(comment.get("body", "") or "")
        lower = body.lower()
        if not any(
            token in lower for token in ("dogfood", "adversarial", "cross-author", "recheck")
        ):
            continue
        reviewer = _infer_model_reviewer_from_text(body)
        if reviewer == "unknown_model_reviewer":
            continue
        author_payload = comment.get("author")
        author = ""
        if isinstance(author_payload, dict):
            author = str(author_payload.get("login", "") or "")
        if author == "github-actions":
            continue
        evidence.append(
            {
                "reviewer_id": reviewer,
                "github_author": author,
                "source": "pr_comment",
                "summary": _first_nonempty_line(body)[:240],
            }
        )
    return evidence[:5]


def _model_review_signals_from_comments(
    comments: list[Any],
    *,
    head_sha: str = "",
    head_committed_at: str = "",
) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        if not _is_comment_grounded_on_head(comment, head_sha, head_committed_at):
            continue
        body = str(comment.get("body", "") or "")
        lower = body.lower()
        if not any(
            token in lower
            for token in (
                "codex review",
                "claude review",
                "grok independent",
                "gemini independent",
                "independent semantic review",
                "independent model review",
                "model-family semantic signal",
            )
        ):
            continue
        reviewer = _infer_model_reviewer_from_text(body)
        if reviewer == "unknown_model_reviewer":
            continue
        author_payload = comment.get("author")
        github_author = ""
        if isinstance(author_payload, dict):
            github_author = str(author_payload.get("login", "") or "")
        if github_author == "github-actions":
            continue
        signals.append(
            {
                "reviewer_id": reviewer,
                "provider": reviewer,
                "source": "pr_comment",
                "summary": _first_nonempty_line(body)[:240],
            }
        )
    return signals[:5]


def _infer_model_reviewer_from_text(text: str) -> str:
    """Infer the reviewing model from a comment body.

    Restricts the substring match to a *structured* marker — the first
    markdown heading line, falling back to the first 200 characters of
    the body when no heading is present.  This avoids false positives
    where a model name appears as a substring deep in the body, for
    example ``codex/some-branch`` in a quoted git command or
    ``claude-mem`` in a file path.  Reviewers conventionally announce
    their identity in the comment's first heading.
    """
    candidate = ""
    for line in str(text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip()
            if candidate:
                break
    if not candidate:
        candidate = str(text)[:200]
    lower = candidate.lower()
    for name in ("claude", "codex", "tesla", "harvey", "factory", "grok", "gemini"):
        if name in lower:
            return name
    return "unknown_model_reviewer"


def _is_comment_grounded_on_head(
    comment: dict[str, Any],
    head_sha: str,
    head_committed_at: str,
) -> bool:
    """Return True when *comment* plausibly reviewed the current head.

    Implements the "grounded in the current head SHA" requirement of
    ``docs/REVIEW_AUTHORITY_PRINCIPLES.md``.  A comment is accepted when
    any of:

    * the caller did not supply head metadata (no-op for legacy paths);
    * the comment was posted at or after the head commit's timestamp;
    * the comment body explicitly cites the head SHA prefix (>= 7 chars).

    A comment whose ``createdAt`` predates the head commit and which
    does not cite the head SHA is treated as stale (it reviewed a
    superseded version of the diff) and excluded from quorum counting.
    """
    if not head_sha or not head_committed_at:
        return True
    body = str(comment.get("body", "") or "")
    head_short = head_sha[:7]
    if head_short and head_short in body:
        return True
    created = str(comment.get("createdAt", "") or "")
    if not created:
        # No timestamp on the comment — fall back to SHA-prefix evidence.
        # We have already established head_sha is set; absence of a
        # citation in body means the reviewer cannot be proven to have
        # seen this head, so the comment is treated as stale.
        return False
    return created >= head_committed_at


def _first_nonempty_line(text: str) -> str:
    for line in str(text).splitlines():
        stripped = line.strip("# ").strip()
        if stripped:
            return stripped
    return ""


def _matches_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def _is_docs_tests_or_status_path(path: str) -> bool:
    return path.startswith(("docs/", "tests/")) or path in {
        "AGENTS.md",
        "CLAUDE.md",
        "README.md",
        "CHANGELOG.md",
        "GA_CHECKLIST.md",
    }


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
    _render_active_auto_handle_alerts()
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
    if packet.model_review_quorum:
        quorum = packet.model_review_quorum
        print()
        print("model review quorum:")
        print(f"  tier: Tier {quorum.get('tier')} ({quorum.get('tier_name', 'unknown')})")
        print(f"  status: {quorum.get('status', 'unknown')}")
        print(f"  verdict: {quorum.get('verdict', 'unknown')}")
        print(f"  admin squash allowed: {quorum.get('admin_squash_allowed', False)}")
        print(
            "  human risk settlement required: "
            f"{quorum.get('requires_human_risk_settlement', False)}"
        )
        print(
            "  signals: "
            f"{len(quorum.get('counted_reviewer_ids') or [])}/"
            f"{quorum.get('required_model_signals', 0)}"
        )
        if quorum.get("counted_reviewer_ids"):
            print(f"  counted reviewers: {', '.join(quorum.get('counted_reviewer_ids') or [])}")
        if quorum.get("unresolved_dissent"):
            print("  unresolved dissent: true")
        for reason in quorum.get("reasons") or []:
            print(f"    - {reason}")
    print()
    print(f"generated at: {packet.generated_at}")
    _render_active_auto_handle_alerts()
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


def _render_merge_authorization_packet(packet: dict[str, Any]) -> None:
    queue = packet.get("queue_pressure") or {}
    print("# Merge authorization packet")
    print(f"generated at: {packet.get('generated_at', '')}")
    print(
        "queue pressure: "
        f"{queue.get('current_open_prs', 0)} open / cap {queue.get('cap', MODEL_REVIEW_QUEUE_CAP)} "
        f"(active={queue.get('active', False)})"
    )
    if queue.get("active"):
        print(
            "new implementation PRs: frozen; only review/dogfood/fix-existing/spec-only work allowed"
        )
    print()
    print("authorization sentence:")
    print(packet.get("authorization_sentence", ""))
    print()

    admin_order = packet.get("admin_squash_order") or []
    human_required = packet.get("human_risk_settlement_required") or []
    not_ready = packet.get("not_ready") or []
    print(f"admin squash order: {', '.join(f'#{n}' for n in admin_order) or '(none)'}")
    print(
        f"human risk settlement required: {', '.join(f'#{n}' for n in human_required) or '(none)'}"
    )
    print(f"not ready: {', '.join(f'#{n}' for n in not_ready) or '(none)'}")
    print()

    for entry in packet.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        print(
            f"#{entry.get('pr_number')} | Tier {entry.get('tier')} | "
            f"{entry.get('status')} | {entry.get('verdict')}"
        )
        print(f"  {entry.get('title', '')}")
        print(f"  head: {entry.get('head_sha', '')}")
        print(f"  checks: {entry.get('checks_summary', '')}")
        print(
            "  evidence: "
            f"{len(entry.get('reviewer_signals') or [])} reviewer signal(s), "
            f"{len(entry.get('dogfood_evidence') or [])} dogfood note(s), "
            f"{len(entry.get('counted_reviewer_ids') or [])} counted reviewer(s)"
        )
        for reason in entry.get("reasons") or []:
            print(f"  - {reason}")
        print()


def _render_baseline_report(
    *,
    measurement: BaselineMeasurement,
    proposal: ThresholdProposal,
) -> None:
    """Print a human-readable empirical-threshold baseline report."""
    print("# Empirical invalidation baseline (gap #6375)")
    print()
    print(f"window:       {measurement.window_start.isoformat()}")
    print(f"           -> {measurement.window_end.isoformat()}")
    print(f"              ({measurement.window_days}d)")
    print()
    print("samples:")
    print(
        f"  human-settled:   {measurement.invalidated_human_settled} invalidated "
        f"/ {measurement.total_human_settled} total"
    )
    print(
        f"  auto-handled:    {measurement.invalidated_auto_handled} invalidated "
        f"/ {measurement.total_auto_handled} total"
    )
    print(
        f"  min required:    {measurement.min_samples_required}  "
        f"(acceptable: {measurement.sample_size_acceptable})"
    )
    print()
    print("rates (with Wilson 95% CI):")
    print(
        "  human baseline:  "
        f"{_fmt_rate(measurement.baseline_human_rate)}  "
        f"[{_fmt_rate(measurement.baseline_human_rate_ci_low)}, "
        f"{_fmt_rate(measurement.baseline_human_rate_ci_high)}]"
    )
    print(
        "  auto-handle:     "
        f"{_fmt_rate(measurement.auto_handle_rate)}  "
        f"[{_fmt_rate(measurement.auto_handle_rate_ci_low)}, "
        f"{_fmt_rate(measurement.auto_handle_rate_ci_high)}]"
    )
    print()
    if measurement.per_class_human:
        print("per-class human breakdown (invalidated/total):")
        for cls, (inv, tot) in sorted(measurement.per_class_human.items()):
            print(f"  - {cls}: {inv}/{tot}")
        print()
    if measurement.per_class_auto:
        print("per-class auto-handle breakdown (invalidated/total):")
        for cls, (inv, tot) in sorted(measurement.per_class_auto.items()):
            print(f"  - {cls}: {inv}/{tot}")
        print()
    if measurement.notes:
        print("notes (data-availability caveats):")
        for key, note in sorted(measurement.notes.items()):
            print(f"  - {key}: {note}")
        print()
    print("proposed threshold:")
    print(f"  value:         {_fmt_rate(proposal.threshold)}")
    print(f"  baseline:      {_fmt_rate(proposal.baseline)}")
    print(f"  safety margin: {proposal.safety_margin:.2f}")
    print(f"  min meaningful rate: {proposal.minimum_meaningful_rate:.4f}")
    print(f"  placeholder:   {proposal.is_placeholder}")
    print(f"  rationale:     {proposal.rationale}")
    print(f"  measured at:   {proposal.measured_at.isoformat()}")
    print()
    print(
        "Note: this command is read-only and advisory. Applying the proposed "
        "threshold or persisting a recalibration receipt is the recalibration "
        "scheduler's job (#6375 step B), not this CLI."
    )


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f} ({value:.2%})"


def _render_active_auto_handle_alerts() -> None:
    try:
        alerts = AutoHandleCalibrationStore().list_active_alerts(limit=3)
    except (OSError, RuntimeError, sqlite3.Error, ValueError, TypeError) as exc:
        print(f"warning: auto-handle calibration unavailable: {exc}", file=sys.stderr)
        return
    if not alerts:
        return
    print()
    print("ACTIVE AUTO-HANDLE DRIFT ALERTS:")
    for alert in alerts:
        current_rate = (
            f"{alert.current_success_rate:.1%}"
            if alert.current_success_rate is not None
            else "unknown"
        )
        print(
            f"  - {alert.auto_handle_path}: {alert.decision_class} "
            f"(success={current_rate}, action={alert.remediation_action})"
        )
