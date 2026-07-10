#!/usr/bin/env python3
"""Re-run externally cancelled PR workflow runs once (M1 re-trigger guardian).

Implements mitigation M1 from docs/governance/PR_RUN_CANCELLATION_DIAGNOSIS.md:
an external cancellation actor intermittently cancels non-required advisory
``pull_request`` runs at the checkout step, leaving terminal-``cancelled``
"failures" that mask real coverage (observed live 2026-07-09: ~6 incidents,
100% of which cleared on manual rerun, including a shard killed at 96%
all-passing).

A cancelled run is re-run ONLY when ALL hold:
- ``conclusion == cancelled`` and the event is a PR event;
- its ``head_sha`` equals the PR's CURRENT head (not superseded);
- no newer run of the same workflow+branch exists (rerun would be moot);
- the PR is open and not draft;
- the run is younger than ``--ttl-hours``;
- ``run_attempt == 1`` — a rerun bumps the attempt counter, so this is the
  stateless once-per-run marker that bounds re-run loops if the external
  actor keeps cancelling.

Dry-run by default; ``--apply`` performs the reruns.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib import error, parse, request

PR_EVENTS = {"pull_request"}


class GitHubApiError(RuntimeError):
    """Raised when GitHub API calls fail."""


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
    ) -> tuple[Any, Any]:
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "aragora-cancelled-run-guardian",
        }
        req = request.Request(url=url, data=body, headers=headers, method=method)
        try:
            resp = request.urlopen(req, timeout=30)
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return parsed, resp
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:200]
            raise GitHubApiError(f"{method} {url} failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise GitHubApiError(f"{method} {url} failed: {exc.reason}") from exc

    def _api(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = f"{self.api_base}{path}"
        if query:
            url = f"{url}?{parse.urlencode(query)}"
        return url

    def get(self, path: str, query: dict[str, Any] | None = None) -> Any:
        data, _ = self._request_json("GET", self._api(path, query))
        return data

    def post(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        data, _ = self._request_json("POST", self._api(path), payload)
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

    def list_recent_workflow_runs(self, max_runs: int) -> list[dict[str, Any]]:
        runs = self.paginate(
            f"/repos/{self.repo}/actions/runs",
            query={"per_page": 100},
            max_pages=max(1, (max_runs + 99) // 100),
        )
        normalized = [r for r in runs if isinstance(r, dict)]
        return normalized[:max_runs]

    def rerun_workflow_run(self, run_id: int) -> tuple[bool, str]:
        try:
            self.post(f"/repos/{self.repo}/actions/runs/{run_id}/rerun")
            return True, "rerun_requested"
        except GitHubApiError as exc:
            return False, str(exc)


def _parse_created_at(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def compute_active_head_map(open_pulls: list[dict[str, Any]]) -> dict[str, str]:
    """Return branch -> head_sha for open, NON-draft PR heads."""
    active: dict[str, str] = {}
    for pr in open_pulls:
        if bool(pr.get("draft")):
            continue
        head = pr.get("head", {})
        branch = str(head.get("ref", "")).strip()
        sha = str(head.get("sha", "")).strip()
        if branch and sha:
            active[branch] = sha
    return active


def compute_reruns(
    runs: list[dict[str, Any]],
    *,
    active_heads: dict[str, str],
    now: datetime,
    ttl_hours: float,
    events: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select cancelled PR runs that deserve exactly one honest rerun."""
    considered_events = events or PR_EVENTS
    cutoff = now - timedelta(hours=ttl_hours)
    newest_by_group: dict[tuple[Any, str], datetime] = {}
    for run in runs:
        created = _parse_created_at(str(run.get("created_at", "")))
        if created is None:
            continue
        group = (run.get("workflow_id"), str(run.get("head_branch", "")).strip())
        if group not in newest_by_group or created > newest_by_group[group]:
            newest_by_group[group] = created

    reruns: list[dict[str, Any]] = []
    for run in runs:
        if str(run.get("event", "")).strip() not in considered_events:
            continue
        if str(run.get("conclusion", "")).strip() != "cancelled":
            continue
        if int(run.get("run_attempt") or 1) != 1:
            continue  # once-per-run marker: a rerun already happened
        branch = str(run.get("head_branch", "")).strip()
        sha = str(run.get("head_sha", "")).strip()
        active_sha = active_heads.get(branch)
        if not active_sha or active_sha != sha:
            continue  # superseded head, closed PR, or draft
        created = _parse_created_at(str(run.get("created_at", "")))
        if created is None or created < cutoff:
            continue
        group = (run.get("workflow_id"), branch)
        newest = newest_by_group.get(group)
        if newest is not None and created < newest:
            continue  # a newer run of this workflow+branch supersedes this one
        reruns.append(
            {
                "run_id": int(run.get("id") or 0),
                "workflow": str(run.get("name", "")).strip(),
                "branch": branch,
                "sha": sha,
                "created_at": str(run.get("created_at", "")),
            }
        )
    return reruns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run externally cancelled PR workflow runs once"
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository in OWNER/REPO format",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=300,
        help="Maximum recent workflow runs to inspect",
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=6.0,
        help="Only rerun cancellations younger than this many hours",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually request reruns (default: dry-run report only)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.repo:
        print("--repo is required (or set GITHUB_REPOSITORY)", file=sys.stderr)
        return 2
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    if not token:
        print("GITHUB_TOKEN (or GH_TOKEN) is required", file=sys.stderr)
        return 2
    client = GitHubClient(args.repo, token)
    active_heads = compute_active_head_map(client.list_open_pulls())
    runs = client.list_recent_workflow_runs(args.max_runs)
    reruns = compute_reruns(
        runs,
        active_heads=active_heads,
        now=datetime.now(timezone.utc),
        ttl_hours=args.ttl_hours,
    )
    for item in reruns:
        if args.apply:
            ok, detail = client.rerun_workflow_run(item["run_id"])
            item["applied"] = ok
            item["detail"] = detail
        else:
            item["applied"] = False
            item["detail"] = "dry-run"
    print(json.dumps({"repo": args.repo, "rerun_count": len(reruns), "reruns": reruns}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
