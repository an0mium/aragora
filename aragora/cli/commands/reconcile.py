"""Read-only reconcile-lane operator commands."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any

from aragora.swarm.auto_merge_green import (
    MAX_AUTO_MERGE_TIER,
    context_from_gh,
    decide_auto_merge,
)

SETTLE_REPORT_VERSION = "reconcile_settle_report.v1"
SETTLE_BUCKETS: tuple[str, ...] = ("mergeable", "parked", "superseded", "needs_human")
PR_VIEW_FIELDS = ",".join(
    (
        "number",
        "title",
        "url",
        "state",
        "mergedAt",
        "headRefName",
        "headRefOid",
        "isDraft",
        "mergeable",
        "mergeStateStatus",
        "statusCheckRollup",
    )
)


def add_reconcile_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "reconcile",
        help="Run reconcile-lane repo cleanup and settlement reports",
        description=(
            "Read-only first slice of the reconcile lane. The settle report reuses "
            "review-queue merge-packet evidence and the auto-merge decision core."
        ),
    )
    sub = parser.add_subparsers(dest="reconcile_command")

    settle = sub.add_parser(
        "settle",
        help="Report exact-head PR settlement buckets without mutating state",
    )
    settle.add_argument(
        "--autonomy",
        default="report",
        choices=("report",),
        help="Autonomy mode. This first slice supports report only.",
    )
    settle.add_argument(
        "--pr",
        action="append",
        default=[],
        help="Specific PR number/ref to include. Repeatable. Defaults to open queue.",
    )
    settle.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Max open PRs to inspect when --pr is not supplied.",
    )
    settle.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override (owner/name). Defaults to current repo context.",
    )
    settle.add_argument(
        "--review-queue-root",
        default=None,
        help="Override the review-queue store root used by merge-packet.",
    )
    settle.add_argument(
        "--execute-reviewers",
        action="store_true",
        help="Pass through to review-queue merge-packet reviewer execution.",
    )
    settle.add_argument(
        "--ignore-own-quorum-check",
        action="store_true",
        help="Diagnostic pass-through to review-queue merge-packet.",
    )
    settle.add_argument("--json", action="store_true", help="Output as JSON.")
    settle.set_defaults(func=cmd_reconcile)
    parser.set_defaults(command="reconcile", func=cmd_reconcile)


def cmd_reconcile(args: argparse.Namespace) -> int:
    command = getattr(args, "reconcile_command", None)
    if command != "settle":
        print("Usage: aragora reconcile {settle}", file=sys.stderr)
        return 2
    return _cmd_settle(args)


def _cmd_settle(args: argparse.Namespace) -> int:
    autonomy = str(getattr(args, "autonomy", "report") or "report")
    if autonomy != "report":
        print(
            "error: only --autonomy report is implemented in this first slice",
            file=sys.stderr,
        )
        return 2

    try:
        packet, views = _load_settle_inputs(args)
        report = build_settle_report(
            packet=packet,
            views=views,
            autonomy=autonomy,
            repo=getattr(args, "repo", None),
        )
    except Exception as exc:  # pragma: no cover - exact exception class comes from gh/review-queue
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _render_settle_report(report)
    return 0


def _load_settle_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    pr_refs = [str(ref).strip() for ref in (getattr(args, "pr", []) or []) if str(ref).strip()]
    repo = getattr(args, "repo", None)
    if pr_refs:
        views: dict[int, dict[str, Any]] = {}
        open_refs: list[str] = []
        for ref in pr_refs:
            view = _fetch_pr_view(ref, repo=repo)
            number = _view_number(view)
            views[number] = view
            if not _view_is_superseded(view):
                open_refs.append(str(number))
        packet = (
            _build_merge_packet(
                pr_refs=open_refs,
                limit=int(getattr(args, "limit", 30) or 30),
                repo=repo,
                review_queue_root=getattr(args, "review_queue_root", None),
                execute_reviewers=bool(getattr(args, "execute_reviewers", False)),
                ignore_own_quorum_check=bool(getattr(args, "ignore_own_quorum_check", False)),
            )
            if open_refs
            else _empty_merge_packet(repo=repo)
        )
        for ref in open_refs:
            view = _fetch_pr_view(ref, repo=repo)
            views[_view_number(view)] = view
        return packet, views

    packet = _build_merge_packet(
        pr_refs=[],
        limit=int(getattr(args, "limit", 30) or 30),
        repo=repo,
        review_queue_root=getattr(args, "review_queue_root", None),
        execute_reviewers=bool(getattr(args, "execute_reviewers", False)),
        ignore_own_quorum_check=bool(getattr(args, "ignore_own_quorum_check", False)),
    )
    views = {}
    for entry in packet.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        number = _entry_number(entry)
        if number <= 0:
            continue
        views[number] = _fetch_pr_view(str(number), repo=repo)
    return packet, views


def build_settle_report(
    *,
    packet: dict[str, Any],
    views: dict[int, dict[str, Any]],
    autonomy: str,
    repo: str | None,
) -> dict[str, Any]:
    entries_by_pr = {
        _entry_number(entry): entry
        for entry in packet.get("entries") or []
        if isinstance(entry, dict) and _entry_number(entry) > 0
    }
    buckets: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in SETTLE_BUCKETS}
    for view in views.values():
        entry = entries_by_pr.get(_view_number(view))
        bucket, record = _classify_settle_record(view, entry)
        buckets[bucket].append(record)

    counts = {bucket: len(buckets[bucket]) for bucket in SETTLE_BUCKETS}
    return {
        "version": SETTLE_REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "autonomy": autonomy,
        "mutated": False,
        "repo": repo,
        "counts": counts,
        "merge_packet": {
            "version": packet.get("version"),
            "generated_at": packet.get("generated_at"),
            "queue_pressure": packet.get("queue_pressure") or {},
            "admin_squash_order": list(packet.get("admin_squash_order") or []),
            "human_risk_settlement_required": list(
                packet.get("human_risk_settlement_required") or []
            ),
            "not_ready": list(packet.get("not_ready") or []),
        },
        **buckets,
    }


def _classify_settle_record(
    view: dict[str, Any],
    entry: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    if _view_is_superseded(view) or (entry and entry.get("status") == "already_merged"):
        return "superseded", _record(
            view, entry, blockers=(), bucket_reason=_superseded_reason(view)
        )

    ctx = context_from_gh(view, entry)
    decision = decide_auto_merge(ctx)
    if decision.should_merge:
        return "mergeable", _record(view, entry, blockers=(), bucket_reason="would_merge")
    if _record_needs_human(entry, decision):
        return (
            "needs_human",
            _record(view, entry, blockers=decision.blockers, bucket_reason="human_required"),
        )
    return "parked", _record(view, entry, blockers=decision.blockers, bucket_reason="not_ready")


def _record(
    view: dict[str, Any],
    entry: dict[str, Any] | None,
    *,
    blockers: tuple[str, ...],
    bucket_reason: str,
) -> dict[str, Any]:
    packet = entry or {}
    return {
        "pr": _view_number(view),
        "title": str(view.get("title") or packet.get("title") or ""),
        "url": str(view.get("url") or packet.get("url") or ""),
        "state": str(view.get("state") or ""),
        "head": str(view.get("headRefOid") or ""),
        "packet_head": str(packet.get("head_sha") or ""),
        "tier": packet.get("tier"),
        "status": packet.get("status"),
        "verdict": packet.get("verdict"),
        "bucket_reason": bucket_reason,
        "blockers": list(blockers),
        "packet_reasons": list(packet.get("reasons") or []),
        "counted_model_families": list(packet.get("counted_model_families") or []),
        "reviewer_signal_count": len(packet.get("reviewer_signals") or []),
        "dogfood_evidence_count": len(packet.get("dogfood_evidence") or []),
    }


def _record_needs_human(entry: dict[str, Any] | None, decision: Any) -> bool:
    if not entry:
        return False
    tier = entry.get("tier")
    try:
        tier_int = int(tier) if tier is not None else None
    except (TypeError, ValueError):
        tier_int = None
    if tier_int is not None and tier_int > MAX_AUTO_MERGE_TIER:
        return True
    if bool(entry.get("requires_human_risk_settlement")):
        return True
    if bool(entry.get("requires_human_preapproval")) and not bool(
        entry.get("human_preapproval_recorded")
    ):
        return True
    if entry.get("status") == "human_risk_settlement_required":
        return True
    return any("human" in blocker.lower() for blocker in decision.blockers)


def _render_settle_report(report: dict[str, Any]) -> None:
    counts = report["counts"]
    print("# Reconcile settle report")
    print(f"autonomy: {report['autonomy']} (read-only; mutated=false)")
    print(
        "counts: "
        f"mergeable={counts['mergeable']} "
        f"parked={counts['parked']} "
        f"superseded={counts['superseded']} "
        f"needs-human={counts['needs_human']}"
    )
    for bucket, label in (
        ("mergeable", "mergeable"),
        ("parked", "parked"),
        ("superseded", "superseded"),
        ("needs_human", "needs-human"),
    ):
        print(f"\n## {label}")
        records = report[bucket]
        if not records:
            print("  (none)")
            continue
        for record in records:
            print(f"  #{record['pr']} {record['title']}")
            print(
                "    "
                f"head={_short(record['head'])} "
                f"tier={record['tier']} "
                f"status={record['status']} "
                f"verdict={record['verdict']}"
            )
            if record["blockers"]:
                print(f"    blockers: {'; '.join(record['blockers'][:4])}")
            if record["packet_reasons"]:
                print(f"    packet: {'; '.join(record['packet_reasons'][:2])}")


def _fetch_pr_view(ref: str, *, repo: str | None) -> dict[str, Any]:
    args = ["pr", "view", str(ref), "--json", PR_VIEW_FIELDS]
    if repo:
        args.extend(["--repo", repo])
    view = _gh_json(args)
    if not isinstance(view, dict):
        raise RuntimeError(f"PR {ref} did not return an object")
    return view


def _build_merge_packet(
    *,
    pr_refs: list[str],
    limit: int,
    repo: str | None,
    review_queue_root: str | None,
    execute_reviewers: bool,
    ignore_own_quorum_check: bool,
) -> dict[str, Any]:
    from aragora.cli.commands.review_queue import _build_merge_authorization_packet

    return _build_merge_authorization_packet(
        pr_refs=pr_refs,
        limit=limit,
        repo_override=repo,
        review_queue_root=review_queue_root,
        execute_reviewers=execute_reviewers,
        ignore_own_quorum_check=ignore_own_quorum_check,
    )


def _gh_json(args: list[str]) -> Any:
    from aragora.cli.commands.review_queue import _gh_json as review_queue_gh_json

    return review_queue_gh_json(args)


def _merge_pr(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError("merge application is intentionally absent from report mode")


def _collect_quorum_evidence_apply(*_args: Any, **_kwargs: Any) -> None:
    raise NotImplementedError("evidence application is intentionally absent from report mode")


def _empty_merge_packet(*, repo: str | None) -> dict[str, Any]:
    return {
        "version": "merge_authorization_packet.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo": repo,
        "queue_pressure": {"current_open_prs": 0, "cap": 0, "active": False},
        "entries": [],
        "admin_squash_order": [],
        "human_risk_settlement_required": [],
        "not_ready": [],
    }


def _view_number(view: dict[str, Any]) -> int:
    try:
        return int(view.get("number") or 0)
    except (TypeError, ValueError):
        return 0


def _entry_number(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("pr_number") or 0)
    except (TypeError, ValueError):
        return 0


def _view_is_superseded(view: dict[str, Any]) -> bool:
    state = str(view.get("state") or "").upper()
    return state in {"MERGED", "CLOSED"} or bool(view.get("mergedAt"))


def _superseded_reason(view: dict[str, Any]) -> str:
    state = str(view.get("state") or "").upper()
    if state == "MERGED" or view.get("mergedAt"):
        return "already_merged"
    if state == "CLOSED":
        return "closed"
    return "not_open"


def _short(value: object) -> str:
    text = str(value or "")
    return text[:12] if text else ""
