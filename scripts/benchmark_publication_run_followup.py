#!/usr/bin/env python3
"""Dry-run-first follow-up helper for Benchmark Truth Publication runs.

This helper is intentionally narrower than the general PR stale-run GC. It
targets recurring trust-loop publication runs on ``main`` and reports stale
queued/pending runs whose recorded branch SHA no longer matches the live branch
head. Apply mode can cancel only the runs it classifies as stale.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pr_stale_run_gc import GitHubApiError, GitHubClient


ACTIVE_RUN_STATUSES = {"queued", "in_progress", "requested", "waiting", "pending"}
DEFAULT_EVENTS = {"schedule", "workflow_dispatch"}
DEFAULT_WORKFLOW = "benchmark-truth-publication.yml"


def _field(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload.get(name)
    return None


def _parse_timestamp(value: str | datetime | None) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _age_minutes(now_dt: datetime, value: str | datetime | None) -> float | None:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return None
    return round(max(0.0, (now_dt - parsed).total_seconds() / 60), 2)


def _normalize_run(run: dict[str, Any], now_dt: datetime) -> dict[str, Any]:
    run_id = _field(run, "databaseId", "id")
    created_at = _field(run, "createdAt", "created_at")
    updated_at = _field(run, "updatedAt", "updated_at")
    return {
        "run_id": int(run_id or 0),
        "event": str(_field(run, "event") or "").strip(),
        "status": str(_field(run, "status") or "").strip(),
        "head_branch": str(_field(run, "headBranch", "head_branch") or "").strip(),
        "head_sha": str(_field(run, "headSha", "head_sha") or "").strip(),
        "age_minutes": _age_minutes(now_dt, created_at),
        "updated_age_minutes": _age_minutes(now_dt, updated_at),
        "url": str(_field(run, "url", "html_url") or "").strip(),
    }


def classify_publication_runs(
    runs: list[dict[str, Any]],
    *,
    branch_heads: dict[str, str],
    now: str | datetime | None = None,
    stale_after_minutes: int = 30,
    events: set[str] | None = None,
    allow_unknown_branch_cancel: bool = False,
) -> list[dict[str, Any]]:
    """Return stale-window publication run actions.

    Missing branch-head truth is report-only by default because it is ambiguous:
    the branch may be gone, or GitHub may have failed to answer the branch
    lookup. Apply mode should only cancel when live branch truth proves staleness
    unless the operator explicitly opts into unknown-branch cancellation.
    """
    now_dt = _parse_timestamp(now) if now is not None else datetime.now(UTC)
    if now_dt is None:
        now_dt = datetime.now(UTC)
    considered_events = events or DEFAULT_EVENTS

    actions: list[dict[str, Any]] = []
    for raw_run in runs:
        run = _normalize_run(raw_run, now_dt)
        if run["event"] not in considered_events:
            continue
        if run["status"] not in ACTIVE_RUN_STATUSES:
            continue
        if run["age_minutes"] is None or run["updated_age_minutes"] is None:
            continue
        if (
            run["age_minutes"] < stale_after_minutes
            or run["updated_age_minutes"] < stale_after_minutes
        ):
            continue

        branch = str(run["head_branch"])
        current_sha = branch_heads.get(branch)
        reason = ""
        action = "cancel"
        if not current_sha:
            reason = "unknown-branch-head"
            action = "cancel" if allow_unknown_branch_cancel else "report"
        elif current_sha != run["head_sha"]:
            reason = "stale-branch-sha"
        else:
            continue

        actions.append(
            {
                "run_id": run["run_id"],
                "action": action,
                "reason": reason,
                "event": run["event"],
                "status": run["status"],
                "head_branch": branch,
                "head_sha": run["head_sha"],
                "current_branch_sha": current_sha or "",
                "age_minutes": run["age_minutes"],
                "updated_age_minutes": run["updated_age_minutes"],
                "url": run["url"],
            }
        )
    return actions


def _list_workflow_runs(
    client: GitHubClient,
    *,
    workflow: str,
    max_runs: int,
) -> list[dict[str, Any]]:
    runs = client.paginate(
        f"/repos/{client.repo}/actions/workflows/{workflow}/runs",
        query={"per_page": 100},
        max_pages=max(1, (max_runs + 99) // 100),
    )
    return [run for run in runs if isinstance(run, dict)][:max_runs]


def _branch_heads_for_runs(client: GitHubClient, runs: list[dict[str, Any]]) -> dict[str, str]:
    branches = {
        str(_field(run, "headBranch", "head_branch") or "").strip()
        for run in runs
        if str(_field(run, "headBranch", "head_branch") or "").strip()
    }
    heads: dict[str, str] = {}
    for branch in sorted(branches):
        encoded_branch = quote(branch, safe="")
        try:
            data = client.get(f"/repos/{client.repo}/branches/{encoded_branch}")
        except GitHubApiError:
            continue
        commit = data.get("commit") if isinstance(data, dict) else {}
        sha = str((commit or {}).get("sha") or "").strip()
        if sha:
            heads[branch] = sha
    return heads


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "synaptent/aragora"),
        help="GitHub repository in OWNER/REPO format",
    )
    parser.add_argument(
        "--workflow",
        default=DEFAULT_WORKFLOW,
        help="Workflow file name or id to inspect",
    )
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument("--stale-after-minutes", type=int, default=30)
    parser.add_argument(
        "--events",
        default="schedule,workflow_dispatch",
        help="Comma-separated event names eligible for stale-run cancellation",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Cancel classified stale runs. Omit for dry-run report only.",
    )
    parser.add_argument(
        "--allow-unknown-branch-cancel",
        action="store_true",
        help=(
            "Allow apply mode to cancel stale-window runs whose live branch head "
            "could not be resolved. By default these are report-only."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_runs < 1:
        print("--max-runs must be >= 1", file=sys.stderr)
        return 1
    token = os.environ.get("GITHUB_TOKEN", "").strip() or os.environ.get("GH_TOKEN", "").strip()
    if not token:
        print("GITHUB_TOKEN or GH_TOKEN is required", file=sys.stderr)
        return 1
    events = {event.strip() for event in str(args.events).split(",") if event.strip()}
    if not events:
        events = set(DEFAULT_EVENTS)

    try:
        client = GitHubClient(repo=str(args.repo), token=token)
        runs = _list_workflow_runs(client, workflow=str(args.workflow), max_runs=args.max_runs)
        branch_heads = _branch_heads_for_runs(client, runs)
        actions = classify_publication_runs(
            runs,
            branch_heads=branch_heads,
            stale_after_minutes=int(args.stale_after_minutes),
            events=events,
            allow_unknown_branch_cancel=bool(args.allow_unknown_branch_cancel),
        )
        cancelled: list[int] = []
        failed: list[dict[str, Any]] = []
        if args.apply:
            for action in actions:
                if action.get("action") != "cancel":
                    continue
                ok, message = client.cancel_workflow_run(int(action["run_id"]))
                if ok:
                    cancelled.append(int(action["run_id"]))
                else:
                    failed.append({"run_id": action["run_id"], "error": message})

        payload = {
            "workflow": str(args.workflow),
            "dry_run": not bool(args.apply),
            "runs_scanned": len(runs),
            "branch_heads": branch_heads,
            "stale_actions": actions,
            "stale_action_count": len(actions),
            "cancel_eligible_count": sum(
                1 for action in actions if action.get("action") == "cancel"
            ),
            "report_only_count": sum(1 for action in actions if action.get("action") != "cancel"),
            "cancelled": cancelled,
            "failed": failed,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(
                "benchmark publication follow-up: "
                f"{len(actions)} stale action(s), dry_run={not bool(args.apply)}"
            )
            for action in actions:
                print(
                    f"- {action['action']} run {action['run_id']} ({action['reason']}; "
                    f"{action['head_branch']} {action['head_sha']} != "
                    f"{action['current_branch_sha'] or 'unknown'})"
                )
        return 1 if failed else 0
    except (GitHubApiError, ValueError) as exc:
        print(f"Benchmark publication follow-up error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
