"""Founder-facing Aragora status report.

This is a read-only transport-compression surface: it answers the repeated
"what is status / what should happen next?" operator question by composing
existing health and merge-packet primitives. It never posts, reruns, approves,
settles, merges, or writes receipts.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc


def add_founder_status_arguments(
    status_parser: argparse.ArgumentParser,
    *,
    default_api_url: str,
) -> None:
    status_parser.add_argument(
        "--founder",
        action="store_true",
        help=(
            "Show a read-only founder ops report: queue pressure, merge blockers, "
            "proof-loop health, latest brief, and one next action."
        ),
    )
    status_parser.add_argument("--json", action="store_true", help="Output founder status as JSON")
    status_parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo slug override for founder status (owner/name).",
    )
    status_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max PRs to inspect for founder status (default: 10).",
    )
    status_parser.add_argument(
        "--repo-root",
        default=None,
        help="Override repo root for founder status local health and brief lookups.",
    )
    status_parser.add_argument(
        "--review-queue-root",
        default=None,
        help="Override .aragora/review-queue root for founder status.",
    )
    status_parser.add_argument(
        "--overnight-root",
        default=None,
        help="Override .aragora/overnight root for founder status health checks.",
    )
    status_parser.add_argument(
        "--automation-receipts-root",
        default=None,
        help="Override .aragora/automation-receipts root for founder status health checks.",
    )
    status_parser.add_argument(
        "--overnight-brief-root",
        default=None,
        help="Override .aragora/overnight-brief root for founder status.",
    )
    status_parser.add_argument(
        "--server",
        "-s",
        default=default_api_url,
        help=f"Server URL to check (default: {default_api_url})",
    )


def _now() -> datetime:
    return datetime.now(tz=UTC)


def _age_hours(path: Path, *, now: datetime) -> float | None:
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None
    return (now - mtime).total_seconds() / 3600.0


def _resolve_repo_root(raw: str | None) -> Path:
    if raw:
        return Path(raw).expanduser().resolve()
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / ".git").exists() or (candidate / ".aragora").exists():
            return candidate.resolve()
    return cwd.resolve()


def _resolve_state_dir(repo_root: Path) -> Path:
    local = repo_root / ".aragora"
    if local.is_dir():
        return local
    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(Path.home() / "Development" / "aragora")
    for candidate in candidates:
        state = candidate if candidate.name == ".aragora" else candidate / ".aragora"
        if state.is_dir():
            return state
    return local


def _newest_file(root: Path) -> Path | None:
    if root.is_file():
        return root
    if not root.exists() or not root.is_dir():
        return None
    newest: tuple[float, Path] | None = None
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if newest is None or mtime > newest[0]:
            newest = (mtime, path)
    return newest[1] if newest is not None else None


def _brief_preview(path: Path, *, max_lines: int = 3) -> str:
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                lines.append(stripped)
                if len(lines) >= max_lines:
                    break
    except OSError:
        return ""
    return " / ".join(lines)


def _latest_brief(state_dir: Path, raw_root: str | None, *, now: datetime) -> dict[str, Any]:
    brief_root = Path(raw_root).expanduser() if raw_root else state_dir / "overnight-brief"
    preferred = brief_root / "latest.md"
    path = preferred if preferred.is_file() else _newest_file(brief_root)
    if path is None:
        return {
            "available": False,
            "root": str(brief_root),
            "path": None,
            "age_hours": None,
            "preview": "",
        }
    return {
        "available": True,
        "root": str(brief_root),
        "path": str(path),
        "age_hours": _age_hours(path, now=now),
        "preview": _brief_preview(path),
    }


def _entry_summary(entry: dict[str, Any]) -> dict[str, Any]:
    raw_reasons = entry.get("reasons")
    reasons: list[Any] = raw_reasons if isinstance(raw_reasons, list) else []
    return {
        "pr_number": entry.get("pr_number"),
        "title": entry.get("title"),
        "url": entry.get("url"),
        "head_sha": entry.get("head_sha"),
        "tier": entry.get("tier"),
        "status": entry.get("status"),
        "verdict": entry.get("verdict"),
        "admin_squash_allowed": bool(entry.get("admin_squash_allowed")),
        "requires_human_risk_settlement": bool(entry.get("requires_human_risk_settlement")),
        "requires_human_preapproval": bool(entry.get("requires_human_preapproval")),
        "unresolved_dissent": bool(entry.get("unresolved_dissent")),
        "checks_summary": entry.get("checks_summary"),
        "counted_model_families": entry.get("counted_model_families") or [],
        "reasons": [str(reason) for reason in reasons[:5]],
    }


def _queue_next_action(
    *,
    queue: dict[str, Any],
    proof_loop: dict[str, Any],
    brief: dict[str, Any],
) -> dict[str, str]:
    if queue.get("transport_status") != "ok":
        return {
            "kind": "repair_transport",
            "summary": "Restore GitHub transport before making queue decisions.",
            "detail": str(queue.get("transport_error") or "merge-packet transport failed"),
        }

    admin_order = queue.get("admin_squash_order") or []
    if admin_order:
        first = admin_order[0]
        return {
            "kind": "settlement_ready",
            "summary": f"Review exact-head merge authority for PR #{first}.",
            "detail": "Use the normal merge gate only; this report is not settlement authority.",
        }

    not_ready = queue.get("not_ready") or []
    not_ready_entries = queue.get("not_ready_entries") or []
    top_entries = queue.get("top_entries") or []
    if not_ready:
        not_ready_set = set(not_ready)
        first_not_ready = next(
            (entry for entry in not_ready_entries if entry.get("pr_number") in not_ready_set),
            None,
        )
        if first_not_ready is None:
            first_not_ready = next(
                (entry for entry in top_entries if entry.get("pr_number") in not_ready_set),
                None,
            )
        if first_not_ready is None:
            first_not_ready = {"pr_number": not_ready[0], "status": "unknown", "reasons": []}
        if first_not_ready:
            pr = first_not_ready.get("pr_number")
            status = first_not_ready.get("status") or "unknown"
            reasons = first_not_ready.get("reasons") or []
            detail = str(reasons[0]) if reasons else "inspect merge-packet blockers"
            return {
                "kind": "queue_blocker",
                "summary": f"Work one bounded blocker on PR #{pr} ({status}).",
                "detail": detail,
            }

    human_required = queue.get("human_risk_settlement_required") or []
    if human_required:
        first = human_required[0]
        return {
            "kind": "human_settlement",
            "summary": f"Record or reject human risk settlement for PR #{first}.",
            "detail": "Human judgment remains explicit; automate only the receipt/status tail.",
        }

    if proof_loop.get("overall_status") in {"stale", "missing"}:
        return {
            "kind": "proof_loop_repair",
            "summary": "Repair stale or missing proof-loop surfaces.",
            "detail": "Run review-queue health for exact stale surfaces.",
        }

    if brief.get("available"):
        return {
            "kind": "read_brief",
            "summary": "Read the latest founder brief, then pick one bounded queue action.",
            "detail": str(brief.get("path") or ""),
        }

    return {
        "kind": "idle",
        "summary": "No queue action was selected from available local evidence.",
        "detail": "Run with --json and inspect transport/proof-loop fields.",
    }


def _not_ready_entries(entries: list[dict[str, Any]], not_ready: list[Any]) -> list[dict[str, Any]]:
    not_ready_set = set(not_ready)
    return [entry for entry in entries if entry.get("pr_number") in not_ready_set]


def _queue_report(
    *,
    limit: int,
    repo: str | None,
    review_queue_root: str | None,
    merge_packet_builder: Any | None = None,
) -> dict[str, Any]:
    from aragora.cli.commands.review_queue_transport import _GhError

    if merge_packet_builder is None:
        from aragora.cli.commands.review_queue import _build_merge_authorization_packet

        merge_packet_builder = _build_merge_authorization_packet

    try:
        packet = merge_packet_builder(
            pr_refs=[],
            limit=limit,
            repo_override=repo,
            review_queue_root=review_queue_root,
            execute_reviewers=False,
            ignore_own_quorum_check=False,
        )
    except (_GhError, RuntimeError, OSError, ValueError) as exc:
        return {
            "transport_status": "blocked",
            "transport_error": f"{type(exc).__name__}: {exc}",
            "queue_pressure": None,
            "status_counts": {},
            "admin_squash_order": [],
            "human_risk_settlement_required": [],
            "not_ready": [],
            "not_ready_entries": [],
            "top_entries": [],
        }

    entries = [
        _entry_summary(entry) for entry in packet.get("entries", []) if isinstance(entry, dict)
    ]
    not_ready = packet.get("not_ready") or []
    counts = Counter(str(entry.get("status") or "unknown") for entry in entries)
    return {
        "transport_status": "ok",
        "transport_error": None,
        "queue_pressure": packet.get("queue_pressure") or {},
        "status_counts": dict(sorted(counts.items())),
        "admin_squash_order": packet.get("admin_squash_order") or [],
        "human_risk_settlement_required": packet.get("human_risk_settlement_required") or [],
        "not_ready": not_ready,
        "not_ready_entries": _not_ready_entries(entries, not_ready),
        "top_entries": entries[: min(limit, 10)],
    }


def gather_founder_status(
    *,
    repo_root: str | None = None,
    repo: str | None = None,
    limit: int = 10,
    review_queue_root: str | None = None,
    overnight_root: str | None = None,
    automation_receipts_root: str | None = None,
    overnight_brief_root: str | None = None,
    health_gatherer: Any | None = None,
    merge_packet_builder: Any | None = None,
) -> dict[str, Any]:
    """Build the read-only founder status report."""
    from aragora.review.health import gather_health

    now = _now()
    resolved_repo = _resolve_repo_root(repo_root)
    state_dir = _resolve_state_dir(resolved_repo)
    health_fn = health_gatherer or gather_health
    health = health_fn(
        repo_root=resolved_repo,
        review_queue_root=Path(review_queue_root).expanduser() if review_queue_root else None,
        overnight_root=Path(overnight_root).expanduser() if overnight_root else None,
        automation_receipts_root=(
            Path(automation_receipts_root).expanduser() if automation_receipts_root else None
        ),
    )
    proof_loop = health.to_dict()
    queue = _queue_report(
        limit=max(1, limit),
        repo=repo,
        review_queue_root=review_queue_root,
        merge_packet_builder=merge_packet_builder,
    )
    brief = _latest_brief(state_dir, overnight_brief_root, now=now)
    next_action = _queue_next_action(queue=queue, proof_loop=proof_loop, brief=brief)
    return {
        "version": "founder_status.v1",
        "generated_at": now.isoformat(),
        "repo_root": str(resolved_repo),
        "state_dir": str(state_dir),
        "queue": queue,
        "proof_loop": proof_loop,
        "latest_brief": brief,
        "next_action": next_action,
    }


def _format_age(age_hours: Any) -> str:
    if age_hours is None:
        return "n/a"
    age = float(age_hours)
    if age < 96:
        return f"{age:.1f}h"
    return f"{age / 24:.1f}d"


def render_founder_status(report: dict[str, Any]) -> str:
    """Render the founder status report for terminal use."""
    queue = report.get("queue") or {}
    proof = report.get("proof_loop") or {}
    brief = report.get("latest_brief") or {}
    next_action = report.get("next_action") or {}
    pressure = queue.get("queue_pressure") or {}

    lines = [
        "Aragora Founder Status",
        f"  generated_at: {report.get('generated_at')}",
        f"  repo_root:    {report.get('repo_root')}",
        (
            "  queue:        "
            f"transport={queue.get('transport_status')} "
            f"open_prs={pressure.get('current_open_prs', 'n/a')} "
            f"cap={pressure.get('cap', 'n/a')} active={pressure.get('active', 'n/a')}"
        ),
        f"  proof_loop:   {str(proof.get('overall_status') or 'unknown').upper()}",
        (
            "  latest_brief: "
            f"{brief.get('path') or '(none)'} age={_format_age(brief.get('age_hours'))}"
        ),
        "",
        f"Next action: {next_action.get('summary', '(none)')}",
    ]
    if next_action.get("detail"):
        lines.append(f"  detail: {next_action['detail']}")

    if queue.get("transport_status") != "ok":
        lines.extend(["", f"Queue transport: {queue.get('transport_error')}"])
    else:
        lines.extend(
            [
                "",
                "Queue summary:",
                f"  status_counts: {queue.get('status_counts') or {}}",
                f"  admin_squash_order: {queue.get('admin_squash_order') or []}",
                (
                    "  human_risk_settlement_required: "
                    f"{queue.get('human_risk_settlement_required') or []}"
                ),
                f"  not_ready: {queue.get('not_ready') or []}",
            ]
        )
        entries = queue.get("top_entries") or []
        if entries:
            lines.append("")
            lines.append("Top PRs:")
            for entry in entries[:5]:
                lines.append(
                    "  "
                    f"#{entry.get('pr_number')} T{entry.get('tier')} "
                    f"{entry.get('status')} | {entry.get('verdict')} | "
                    f"{entry.get('checks_summary')}"
                )
                reasons = entry.get("reasons") or []
                if reasons:
                    lines.append(f"    - {reasons[0]}")

    stale = [
        surface
        for surface in proof.get("surfaces", [])
        if isinstance(surface, dict) and surface.get("status") in {"stale", "missing"}
    ]
    if stale:
        lines.append("")
        lines.append("Proof-loop attention:")
        for surface in stale[:5]:
            lines.append(
                "  "
                f"{surface.get('name')}: {surface.get('status')} "
                f"age={_format_age(surface.get('age_hours'))} "
                f"{surface.get('detail') or ''}"
            )

    preview = brief.get("preview")
    if preview:
        lines.extend(["", f"Brief preview: {preview}"])

    return "\n".join(lines)


def cmd_founder_status(args: argparse.Namespace) -> int:
    report = gather_founder_status(
        repo_root=getattr(args, "repo_root", None),
        repo=getattr(args, "repo", None),
        limit=int(getattr(args, "limit", 10) or 10),
        review_queue_root=getattr(args, "review_queue_root", None),
        overnight_root=getattr(args, "overnight_root", None),
        automation_receipts_root=getattr(args, "automation_receipts_root", None),
        overnight_brief_root=getattr(args, "overnight_brief_root", None),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_founder_status(report))
    return 0


__all__ = [
    "add_founder_status_arguments",
    "cmd_founder_status",
    "gather_founder_status",
    "render_founder_status",
]
