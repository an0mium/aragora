#!/usr/bin/env python3
"""Re-trigger PR workflow runs that were cancelled but not superseded.

Companion to ``scripts/pr_stale_run_gc.py``. The GC *cancels* stale PR runs;
this tool *re-runs* PR runs that were cancelled while still current, so a
freshly-opened PR's advisory checks recover instead of staying red.

See ``docs/governance/PR_RUN_CANCELLATION_DIAGNOSIS.md``: on a freshly-opened
PR a subset of advisory ``pull_request`` runs is cancelled at checkout by an
external actor, even though the run is for the PR's current head and nothing
superseded it. This tool detects exactly that case and re-runs it once.

A run is *re-triggerable* when all hold:
- its event is a PR event (``pull_request`` / ``pull_request_target``);
- its conclusion is ``cancelled`` (a completed-but-cancelled run);
- its branch maps to an open, non-draft PR head (draft PRs are skipped);
- its head SHA equals that PR's current head (not superseded by a new push);
- no newer re-run of the same workflow+branch+head SHA via a PR event exists
  (an unrelated ``push``/``workflow_dispatch`` or different-SHA run does not count);
- it was created within a short TTL (stale cancellations are left alone);
- it has not already been re-run (``run_attempt`` guard + optional marker file).

The only privileged action is ``POST /actions/runs/{id}/rerun`` (equivalent to
``gh run rerun``), gated behind ``--apply``. Default is a dry run that prints a
JSON summary ``{scanned, candidates, eligible, reasons, ...}``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, parse, request


PR_EVENTS = {"pull_request", "pull_request_target"}
APP_TOKEN_RERUN_PERMISSION_MESSAGE = "Resource not accessible by integration"
APP_TOKEN_RERUN_PERMISSION_RESULT = (
    "app-token-rerun-permission-denied: Resource not accessible by integration"
)
OPERATOR_ACTION_EXIT = 2


class GitHubApiError(RuntimeError):
    """Raised when GitHub API calls fail."""


def _resolve_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token
    completed = subprocess.run(
        ["gh", "auth", "token"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


class GitHubClient:
    """Small GitHub API client using urllib and GITHUB_TOKEN."""

    def __init__(self, repo: str, token: str) -> None:
        if "/" not in repo:
            raise ValueError(f"Invalid repo format '{repo}', expected OWNER/REPO")
        self.repo = repo
        self.token = token
        self.api_base = "https://api.github.com"

    def _request_json(
        self,
        method: str,
        url: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[Any, request.addinfourl]:
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "aragora-retrigger-cancelled-pr-runs",
        }
        req = request.Request(url=url, data=body, headers=headers, method=method)
        try:
            resp = request.urlopen(req, timeout=30)
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return parsed, resp
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise GitHubApiError(
                f"GitHub API {method} {url} failed: {exc.code} {exc.reason}\n{details}"
            ) from exc

    def _api(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = f"{self.api_base}{path}"
        if query:
            url += "?" + parse.urlencode(query, doseq=True)
        return url

    def get(self, path: str, query: dict[str, Any] | None = None) -> Any:
        data, _ = self._request_json("GET", self._api(path, query))
        return data

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        data, _ = self._request_json("POST", self._api(path), payload=payload)
        return data

    def paginate(
        self,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        max_pages: int = 10,
    ) -> list[Any]:
        items: list[Any] = []
        current_query = dict(query or {})
        current_query.setdefault("per_page", 100)
        current_query.setdefault("page", 1)

        pages = 0
        while pages < max_pages:
            pages += 1
            data, response = self._request_json("GET", self._api(path, current_query))
            if isinstance(data, list):
                page_items = data
            elif isinstance(data, dict) and "workflow_runs" in data:
                page_items = data["workflow_runs"]
            else:
                raise GitHubApiError(
                    f"Expected list-like response for paginated endpoint {path}, got {type(data)}"
                )
            if not page_items:
                break
            items.extend(page_items)

            link_header = response.headers.get("Link", "")
            if 'rel="next"' not in link_header:
                break
            current_query["page"] = int(current_query["page"]) + 1
        return items

    def list_open_pulls(self) -> list[dict[str, Any]]:
        pulls = self.paginate(
            f"/repos/{self.repo}/pulls",
            query={"state": "open", "per_page": 100},
            max_pages=5,
        )
        return [p for p in pulls if isinstance(p, dict)]

    def get_pull(self, pr_number: int) -> dict[str, Any]:
        pull = self.get(f"/repos/{self.repo}/pulls/{pr_number}")
        if not isinstance(pull, dict):
            raise GitHubApiError(f"Expected PR object for #{pr_number}, got {type(pull)}")
        return pull

    def list_recent_workflow_runs(
        self,
        max_runs: int,
        *,
        branch: str | None = None,
        event: str | None = None,
    ) -> list[dict[str, Any]]:
        query: dict[str, Any] = {"per_page": 100}
        if branch:
            query["branch"] = branch
        if event:
            query["event"] = event
        runs = self.paginate(
            f"/repos/{self.repo}/actions/runs",
            query=query,
            max_pages=max(1, (max_runs + 99) // 100),
        )
        normalized = [r for r in runs if isinstance(r, dict)]
        return normalized[:max_runs]

    def rerun_workflow_run(self, run_id: int) -> tuple[bool, str]:
        try:
            self.post(f"/repos/{self.repo}/actions/runs/{run_id}/rerun")
            return True, "rerun_requested"
        except GitHubApiError as exc:
            message = str(exc)
            if _is_app_token_rerun_permission_failure(message):
                return False, APP_TOKEN_RERUN_PERMISSION_RESULT
            if "403" in message:
                return False, "forbidden"
            return False, message


def _field(run: dict[str, Any], *names: str, default: str = "") -> Any:
    """Return the first present non-empty field (handles REST/CLI casing)."""
    for name in names:
        value = run.get(name)
        if value not in (None, ""):
            return value
    return default


def _is_app_token_rerun_permission_failure(message: str) -> bool:
    return "403" in message and APP_TOKEN_RERUN_PERMISSION_MESSAGE in message


def _parse_iso(ts: str) -> datetime | None:
    text = str(ts).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _run_sort_key(run: dict[str, Any]) -> tuple[float, int, int]:
    created = _parse_iso(str(_field(run, "created_at", "createdAt")))
    created_ts = created.timestamp() if created else 0.0
    run_number = int(_field(run, "run_number", "runNumber", default=0) or 0)
    run_id = int(_field(run, "id", "databaseId", default=0) or 0)
    return (created_ts, run_number, run_id)


def _run_branch(run: dict[str, Any]) -> str:
    return str(_field(run, "head_branch", "headBranch")).strip()


def _run_workflow_key(run: dict[str, Any]) -> Any:
    return _field(run, "workflow_id", "workflowId") or _field(run, "name", default="")


def _run_matches_pr(run: dict[str, Any], pr_number: int) -> bool:
    pulls = run.get("pull_requests") or run.get("pullRequests") or []
    if not isinstance(pulls, list):
        return False
    for pull in pulls:
        if not isinstance(pull, dict):
            continue
        try:
            number = int(pull.get("number", 0) or 0)
        except (TypeError, ValueError):
            continue
        if number == pr_number:
            return True
    return False


def _has_newer_sibling(
    run: dict[str, Any],
    runs: list[dict[str, Any]],
    *,
    pr_events: set[str] | None = None,
) -> bool:
    """True if a genuine newer re-run of the *same* PR check context exists.

    A sibling only supersedes ``run`` when it targets the same workflow, branch,
    and head SHA via a pull-request event. Unrelated ``push`` /
    ``workflow_dispatch`` runs, or runs for a different head SHA on the same
    branch, must not suppress re-running a still-current cancelled PR run.
    """
    wf = _run_workflow_key(run)
    branch = _run_branch(run)
    sha = str(_field(run, "head_sha", "headSha")).strip()
    key = _run_sort_key(run)
    run_id = int(_field(run, "id", "databaseId", default=0) or 0)
    for other in runs:
        other_id = int(_field(other, "id", "databaseId", default=0) or 0)
        if other_id == run_id:
            continue
        if _run_workflow_key(other) != wf or _run_branch(other) != branch:
            continue
        if str(_field(other, "head_sha", "headSha")).strip() != sha:
            continue
        if pr_events is not None and str(_field(other, "event")).strip() not in pr_events:
            continue
        if _run_sort_key(other) > key:
            return True
    return False


def compute_retriggerable_runs(
    runs: list[dict[str, Any]],
    *,
    active_heads: dict[str, str],
    cancel_events: set[str],
    now: datetime,
    ttl_minutes: int,
    already_retriggered: set[int] | None = None,
    max_attempts: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    """Classify cancelled PR runs into re-triggerable vs skipped.

    Returns ``(eligible, reasons, candidates)`` where ``eligible`` is a list of
    run descriptors to re-run, ``reasons`` counts skip reasons among cancelled
    PR runs, and ``candidates`` is the number of cancelled PR runs examined.
    """
    marker = already_retriggered or set()
    ttl_seconds = max(0, ttl_minutes) * 60
    eligible: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    candidates = 0

    def mark(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for run in runs:
        event_name = str(_field(run, "event")).strip()
        if event_name not in cancel_events:
            continue
        conclusion = str(_field(run, "conclusion")).strip().lower()
        status = str(_field(run, "status")).strip().lower()
        if conclusion != "cancelled" or status != "completed":
            continue

        # From here on the run is a completed-but-cancelled PR run: a candidate.
        candidates += 1
        run_id = int(_field(run, "id", "databaseId", default=0) or 0)
        branch = _run_branch(run)
        sha = str(_field(run, "head_sha", "headSha")).strip()
        run_attempt = int(_field(run, "run_attempt", "runAttempt", default=1) or 1)

        if not branch or run_id <= 0:
            mark("missing-branch")
            continue
        active_sha = active_heads.get(branch)
        if active_sha is None:
            mark("draft-or-closed")
            continue
        if active_sha != sha:
            mark("superseded-sha")
            continue
        if _has_newer_sibling(run, runs, pr_events=cancel_events):
            mark("superseded-by-newer-run")
            continue

        created = _parse_iso(str(_field(run, "created_at", "createdAt")))
        if created is None:
            mark("bad-timestamp")
            continue
        if (now - created).total_seconds() > ttl_seconds:
            mark("ttl-expired")
            continue

        if run_id in marker:
            mark("already-retriggered")
            continue
        if run_attempt >= max_attempts:
            mark("max-attempts")
            continue

        eligible.append(
            {
                "run_id": run_id,
                "workflow": str(_field(run, "name", default="")),
                "branch": branch,
                "sha": sha,
                "run_attempt": run_attempt,
                "rerun_command": f"gh run rerun {run_id}",
                "human_rerun_command": f"gh run rerun {run_id}",
            }
        )

    return eligible, reasons, candidates


def load_marker(path: str) -> dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def prune_marker(data: dict[str, str], *, now: datetime, retention_hours: int) -> dict[str, str]:
    if retention_hours <= 0:
        return dict(data)
    cutoff = retention_hours * 3600
    pruned: dict[str, str] = {}
    for run_id, ts in data.items():
        when = _parse_iso(ts)
        if when is None:
            continue
        if (now - when).total_seconds() <= cutoff:
            pruned[run_id] = ts
    return pruned


def save_marker(path: str, data: dict[str, str]) -> None:
    if not path:
        return
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _safe_filename_fragment(value: str) -> str:
    fragment = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return fragment.strip("-") or "all"


def _prune_receipts(path: Path, *, now: datetime, retention_hours: int) -> None:
    if retention_hours <= 0 or not path.exists():
        return
    cutoff = now.timestamp() - (retention_hours * 3600)
    for child in path.glob("RETRIGGER_CANCELLED_RECEIPT_*.json"):
        try:
            if child.stat().st_mtime < cutoff:
                child.unlink()
        except OSError:
            continue


def _write_receipt(
    *,
    receipt_dir: str,
    now: datetime,
    repo: str,
    scope: str,
    summary: dict[str, Any],
    retention_hours: int,
) -> str:
    path = Path(receipt_dir)
    path.mkdir(parents=True, exist_ok=True)
    _prune_receipts(path, now=now, retention_hours=retention_hours)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    filename = (
        f"RETRIGGER_CANCELLED_RECEIPT_{timestamp}_pid{os.getpid()}_"
        f"{_safe_filename_fragment(scope)}.json"
    )
    receipt_path = path / filename
    receipt = {
        "schema": "retrigger-cancelled-pr-runs-receipt/v1",
        "generated_at": now.isoformat(),
        "repo": repo,
        "scope": scope,
        "dry_run": bool(summary.get("dry_run", True)),
        "scanned": int(summary.get("scanned", 0) or 0),
        "candidates": int(summary.get("candidates", 0) or 0),
        "eligible": int(summary.get("eligible", 0) or 0),
        "eligible_run_ids": [int(item["run_id"]) for item in summary.get("eligible_runs", [])],
        "applied": int(summary.get("applied", 0) or 0),
        "apply_failed": int(summary.get("apply_failed", 0) or 0),
        "rerun_run_ids": [
            int(item["run_id"]) for item in summary.get("rerun_results", []) if item.get("ok")
        ],
        "operator_action_required": bool(summary.get("operator_action_required", False)),
        "permission_denied_run_ids": [
            int(item["run_id"]) for item in summary.get("permission_denied_reruns", [])
        ],
        "human_rerun_commands": list(summary.get("human_rerun_commands", [])),
        "operator_packet": str(summary.get("operator_packet", "") or ""),
        "head_shas": sorted(
            {
                str(item.get("sha", "")).strip()
                for item in summary.get("eligible_runs", [])
                if str(item.get("sha", "")).strip()
            }
        ),
        "summary": summary,
    }
    with receipt_path.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return str(receipt_path)


def _human_rerun_items(
    eligible_runs: list[dict[str, Any]],
    rerun_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible_by_id = {int(item["run_id"]): item for item in eligible_runs}
    items: list[dict[str, Any]] = []
    for result in rerun_results:
        if result.get("ok"):
            continue
        message = str(result.get("message", ""))
        if APP_TOKEN_RERUN_PERMISSION_RESULT not in message:
            continue
        run_id = int(result["run_id"])
        run = eligible_by_id.get(run_id, {"run_id": run_id})
        item = dict(run)
        item["run_id"] = run_id
        item["failure"] = APP_TOKEN_RERUN_PERMISSION_MESSAGE
        item["human_rerun_command"] = str(
            item.get("human_rerun_command") or f"gh run rerun {run_id} --failed"
        )
        items.append(item)
    return items


def _github_run_url(repo: str, run_id: int) -> str:
    return f"https://github.com/{repo}/actions/runs/{run_id}"


def _write_operator_rerun_packet(
    *,
    packet_dir: str,
    now: datetime,
    repo: str,
    scope: str,
    human_reruns: list[dict[str, Any]],
    receipt_path: str = "",
) -> str:
    path = Path(packet_dir)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    packet_path = path / (
        f"HUMAN_RERUN_PACKET_{timestamp}_pid{os.getpid()}_{_safe_filename_fragment(scope)}.md"
    )

    lines = [
        "# Cancelled Run Human Rerun Packet",
        "",
        f"Generated: {now.isoformat()}",
        "",
        "Purpose: surface exact human/operator reruns after the GitHub App token "
        f"failed with `{APP_TOKEN_RERUN_PERMISSION_MESSAGE}`.",
        "",
        "This packet does not edit workflow files, mutate PR state, merge, settle, "
        "post evidence, or rerun checks by itself.",
        "",
        "## Pending Reruns",
        "",
        "| Priority | Link | Requested Action | Operator Command |",
        "| --- | --- | --- | --- |",
    ]
    for item in human_reruns:
        run_id = int(item["run_id"])
        command = str(item.get("human_rerun_command") or f"gh run rerun {run_id} --failed")
        workflow = str(item.get("workflow", "") or "workflow run")
        branch = str(item.get("branch", "") or "unknown-branch")
        sha = str(item.get("sha", "") or "unknown-sha")
        requested = (
            f"Run exactly `{command}` with human/operator credentials. "
            f"Proof: `{workflow}` on `{branch}` at `{sha}` was selected as a current "
            f"cancelled run, but the App-token rerun attempt failed with "
            f"`{APP_TOKEN_RERUN_PERMISSION_MESSAGE}`."
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    "P1",
                    _github_run_url(repo, run_id),
                    requested,
                    f"`{command}`",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Source",
            "",
            f"- Repository: `{repo}`",
            f"- Scope: `{scope}`",
        ]
    )
    if receipt_path:
        lines.append(f"- JSON receipt: `{receipt_path}`")
    lines.append("")
    packet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(packet_path)


def compute_active_head_map(
    open_pulls: list[dict[str, Any]],
    *,
    keep_draft_runs: bool,
) -> dict[str, str]:
    """Return branch -> head_sha for open PR heads (draft excluded by default)."""
    active: dict[str, str] = {}
    for pr in open_pulls:
        if bool(pr.get("draft")) and not keep_draft_runs:
            continue
        head = pr.get("head", {})
        branch = str(head.get("ref", "")).strip()
        sha = str(head.get("sha", "")).strip()
        if branch and sha:
            active[branch] = sha
    return active


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-trigger cancelled, non-superseded PR runs")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository in OWNER/REPO format",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="Scope scanning to one pull request's branch/head",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=300,
        help="Maximum recent workflow runs to inspect",
    )
    parser.add_argument(
        "--ttl-minutes",
        type=int,
        default=60,
        help="Only re-trigger runs created within this many minutes",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Skip runs whose run_attempt is >= this value (loop guard)",
    )
    parser.add_argument(
        "--events",
        default="pull_request,pull_request_target",
        help="Comma-separated run events to consider",
    )
    parser.add_argument(
        "--keep-draft-runs",
        action="store_true",
        help="Treat draft PR branches as active and re-trigger their runs",
    )
    parser.add_argument(
        "--marker-file",
        default=os.environ.get("RETRIGGER_MARKER_FILE", ""),
        help="Optional path to a JSON loop-guard marker (run_id -> ISO timestamp)",
    )
    parser.add_argument(
        "--marker-retention-hours",
        type=int,
        default=24,
        help="Prune marker entries older than this many hours",
    )
    parser.add_argument(
        "--receipt-dir",
        default=os.environ.get("RETRIGGER_RECEIPT_DIR", ".aragora/retrigger_cancelled/receipts"),
        help="Directory for per-invocation JSON receipts",
    )
    parser.add_argument(
        "--receipt-retention-hours",
        type=int,
        default=168,
        help="Prune retrigger receipts older than this many hours (<=0 disables pruning)",
    )
    parser.add_argument(
        "--operator-packet-dir",
        default=os.environ.get(
            "RETRIGGER_OPERATOR_PACKET_DIR", ".aragora/retrigger_cancelled/operator_packets"
        ),
        help="Directory for human rerun packets when App-token reruns are permission-denied",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually re-run eligible runs (default: dry run)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.repo:
        print("--repo is required (or set GITHUB_REPOSITORY)", file=sys.stderr)
        return 1
    token = _resolve_github_token()
    if not token:
        print(
            "GITHUB_TOKEN is required, or authenticate gh so `gh auth token` works", file=sys.stderr
        )
        return 1
    if args.max_runs < 1:
        print("--max-runs must be >= 1", file=sys.stderr)
        return 1

    cancel_events = {e.strip() for e in args.events.split(",") if e.strip()}
    if not cancel_events:
        cancel_events = set(PR_EVENTS)

    now = datetime.now(timezone.utc)
    marker_data = prune_marker(
        load_marker(args.marker_file), now=now, retention_hours=args.marker_retention_hours
    )
    already = {int(k) for k in marker_data if str(k).isdigit()}

    try:
        client = GitHubClient(repo=args.repo, token=token)
        scoped_pr: dict[str, Any] | None = None
        if args.pr is not None:
            if args.pr <= 0:
                print("--pr must be a positive integer", file=sys.stderr)
                return 1
            scoped_pr = client.get_pull(int(args.pr))
            open_pulls = [scoped_pr] if str(scoped_pr.get("state")) == "open" else []
        else:
            open_pulls = client.list_open_pulls()
        active_heads = compute_active_head_map(
            open_pulls, keep_draft_runs=bool(args.keep_draft_runs)
        )
        runs: list[dict[str, Any]]
        scoped_branch = ""
        scoped_sha = ""
        if scoped_pr is not None:
            head = scoped_pr.get("head", {})
            scoped_branch = str(head.get("ref", "")).strip()
            scoped_sha = str(head.get("sha", "")).strip()
            runs_by_id: dict[int, dict[str, Any]] = {}
            for event_name in sorted(cancel_events):
                for run in client.list_recent_workflow_runs(
                    max_runs=args.max_runs,
                    branch=scoped_branch,
                    event=event_name,
                ):
                    run_id = int(_field(run, "id", "databaseId", default=0) or 0)
                    if run_id > 0 and _run_matches_pr(run, int(args.pr)):
                        runs_by_id[run_id] = run
            runs = sorted(runs_by_id.values(), key=_run_sort_key, reverse=True)
        else:
            runs = client.list_recent_workflow_runs(max_runs=args.max_runs)
        eligible, reasons, candidates = compute_retriggerable_runs(
            runs,
            active_heads=active_heads,
            cancel_events=cancel_events,
            now=now,
            ttl_minutes=args.ttl_minutes,
            already_retriggered=already,
            max_attempts=args.max_attempts,
        )

        applied = 0
        apply_failed = 0
        rerun_results: list[dict[str, Any]] = []
        if args.apply:
            for item in eligible:
                run_id = int(item["run_id"])
                ok, msg = client.rerun_workflow_run(run_id)
                rerun_results.append({"run_id": run_id, "ok": ok, "message": msg})
                if ok:
                    applied += 1
                    marker_data[str(item["run_id"])] = now.isoformat()
                else:
                    apply_failed += 1
            if args.marker_file:
                save_marker(args.marker_file, marker_data)

        human_reruns = _human_rerun_items(eligible, rerun_results)

        summary = {
            "open_prs_total": len(open_pulls),
            "active_heads_total": len(active_heads),
            "scanned": len(runs),
            "candidates": candidates,
            "eligible": len(eligible),
            "eligible_runs": eligible,
            "reasons": reasons,
            "dry_run": not args.apply,
            "applied": applied,
            "apply_failed": apply_failed,
            "rerun_results": rerun_results,
            "ttl_minutes": args.ttl_minutes,
            "max_attempts": args.max_attempts,
            "pr": int(args.pr) if args.pr is not None else None,
            "scope": f"pr-{args.pr}" if args.pr is not None else "all-open-prs",
            "scoped_branch": scoped_branch,
            "scoped_sha": scoped_sha,
            "operator_action_required": bool(human_reruns),
            "permission_denied_reruns": human_reruns,
            "human_rerun_commands": [
                str(item.get("human_rerun_command", "")) for item in human_reruns
            ],
            "operator_packet": "",
            "operator_packet_error": "",
        }
        operator_packet_error = ""
        if human_reruns:
            try:
                packet_path = _write_operator_rerun_packet(
                    packet_dir=args.operator_packet_dir,
                    now=now,
                    repo=args.repo,
                    scope=str(summary["scope"]),
                    human_reruns=human_reruns,
                )
                summary["operator_packet"] = packet_path
            except OSError as exc:
                operator_packet_error = str(exc)
                summary["operator_packet_error"] = operator_packet_error
        try:
            receipt_path = _write_receipt(
                receipt_dir=args.receipt_dir,
                now=now,
                repo=args.repo,
                scope=str(summary["scope"]),
                summary=summary,
                retention_hours=int(args.receipt_retention_hours),
            )
            summary["receipt"] = receipt_path
            summary["receipt_error"] = ""
        except OSError as exc:
            summary["receipt"] = ""
            summary["receipt_error"] = str(exc)
        print(json.dumps(summary))
        if human_reruns:
            if operator_packet_error or summary.get("receipt_error"):
                return 1
            return OPERATOR_ACTION_EXIT
        return 0
    except (GitHubApiError, ValueError) as exc:
        print(f"Re-trigger error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
