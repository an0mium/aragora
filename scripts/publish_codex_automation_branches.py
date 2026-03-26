#!/usr/bin/env python3
"""Publish committed codex automation branches from a normal user shell.

Codex desktop automations can run in a restricted sandbox that still allows
local git/test work but blocks GitHub network access. This helper scans recent
local ``codex/*`` branches, skips anything dirty or already represented by an
open PR, and optionally pushes branches plus opens PRs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
DEFAULT_SINCE_HOURS = 72
ACTIVE_SESSION_FILES = (
    ".claude-session-active",
    ".codex_session_active",
    ".nomic-session-active",
)


@dataclass(frozen=True)
class BranchSnapshot:
    branch: str
    upstream: str | None
    head_sha: str
    committed_at: datetime
    subject: str
    unique_commit_count: int


@dataclass(frozen=True)
class WorktreeSnapshot:
    path: str
    branch: str | None
    detached: bool
    dirty: bool
    active_session: bool


@dataclass(frozen=True)
class PublishDecision:
    branch: str
    eligible: bool
    reason: str
    subject: str
    head_sha: str
    unique_commit_count: int
    upstream: str | None
    committed_at: str
    worktree_paths: list[str]


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=check,
    )


def _parse_git_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def _repo_root(path: Path) -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "not a git repository")
    return Path(proc.stdout.strip()).resolve()


def _branch_subject(repo_root: Path, branch: str) -> str:
    proc = _run(["git", "log", "-1", "--pretty=%s", branch], cwd=repo_root)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _branch_unique_commit_count(repo_root: Path, base: str, branch: str) -> int:
    proc = _run(["git", "rev-list", "--count", f"{base}..{branch}"], cwd=repo_root)
    if proc.returncode != 0:
        return 0
    try:
        return int(proc.stdout.strip() or "0")
    except ValueError:
        return 0


def _branch_is_merged(repo_root: Path, base: str, branch: str) -> bool:
    proc = _run(["git", "merge-base", "--is-ancestor", branch, base], cwd=repo_root)
    return proc.returncode == 0


def _local_codex_branches(repo_root: Path) -> list[BranchSnapshot]:
    proc = _run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname:short)|%(upstream:short)|%(objectname:short)|%(committerdate:iso8601)|%(subject)",
            "refs/heads/codex/*",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to enumerate codex branches")

    snapshots: list[BranchSnapshot] = []
    for raw_line in proc.stdout.splitlines():
        if not raw_line.strip():
            continue
        branch, upstream, sha, committed_at, subject = raw_line.split("|", 4)
        snapshots.append(
            BranchSnapshot(
                branch=branch,
                upstream=upstream or None,
                head_sha=sha,
                committed_at=_parse_git_dt(committed_at),
                subject=subject,
                unique_commit_count=0,
            )
        )
    return snapshots


def _has_active_session(path: Path) -> bool:
    return any((path / name).exists() for name in ACTIVE_SESSION_FILES)


def _worktree_is_dirty(path: Path) -> bool:
    proc = _run(["git", "status", "--porcelain"], cwd=path)
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def _list_worktrees(repo_root: Path) -> list[WorktreeSnapshot]:
    proc = _run(["git", "worktree", "list", "--porcelain"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to list worktrees")

    snapshots: list[WorktreeSnapshot] = []
    current_path: Path | None = None
    current_branch: str | None = None
    detached = False

    def flush() -> None:
        if current_path is None:
            return
        snapshots.append(
            WorktreeSnapshot(
                path=str(current_path),
                branch=current_branch,
                detached=detached,
                dirty=_worktree_is_dirty(current_path),
                active_session=_has_active_session(current_path),
            )
        )

    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            flush()
            current_path = Path(line.removeprefix("worktree ").strip()).resolve()
            current_branch = None
            detached = False
        elif line.startswith("branch "):
            ref = line.removeprefix("branch ").strip()
            current_branch = ref.removeprefix("refs/heads/")
        elif line == "detached":
            detached = True

    flush()
    return snapshots


def _open_pr_heads(repo_root: Path, repo: str) -> set[str]:
    proc = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "200",
            "--json",
            "headRefName",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to list open PRs")
    payload = json.loads(proc.stdout or "[]")
    return {
        item["headRefName"]
        for item in payload
        if isinstance(item, dict) and isinstance(item.get("headRefName"), str)
    }


def _branches_with_pr_history(repo_root: Path, repo: str, branches: list[str]) -> set[str]:
    historical: set[str] = set()
    for branch in branches:
        proc = _run(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "all",
                "--head",
                branch,
                "--limit",
                "1",
                "--json",
                "number",
            ],
            cwd=repo_root,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                proc.stderr.strip()
                or proc.stdout.strip()
                or f"failed to query PR history for {branch}"
            )
        payload = json.loads(proc.stdout or "[]")
        if payload:
            historical.add(branch)
    return historical


def select_publishable_branches(
    branches: list[BranchSnapshot],
    worktrees: list[WorktreeSnapshot],
    open_pr_heads: set[str],
    *,
    cutoff: datetime,
    base: str,
    is_merged: dict[str, bool] | None = None,
    historical_pr_branches: set[str] | None = None,
) -> list[PublishDecision]:
    worktrees_by_branch: dict[str, list[WorktreeSnapshot]] = {}
    for worktree in worktrees:
        if worktree.branch:
            worktrees_by_branch.setdefault(worktree.branch, []).append(worktree)

    merged_lookup = is_merged or {}
    historical_lookup = historical_pr_branches or set()
    decisions: list[PublishDecision] = []

    for branch in sorted(branches, key=lambda item: item.committed_at, reverse=True):
        attached = worktrees_by_branch.get(branch.branch, [])
        reason = "eligible"

        if merged_lookup.get(branch.branch, False):
            reason = "already_merged"
        elif branch.unique_commit_count <= 0:
            reason = "no_unique_commits"
        elif branch.committed_at < cutoff:
            reason = "older_than_cutoff"
        elif branch.branch in open_pr_heads:
            reason = "open_pr_exists"
        elif branch.branch in historical_lookup:
            reason = "historical_pr_exists"
        elif any(worktree.active_session for worktree in attached):
            reason = "active_session"
        elif any(worktree.dirty for worktree in attached):
            reason = "dirty_worktree"

        decisions.append(
            PublishDecision(
                branch=branch.branch,
                eligible=reason == "eligible",
                reason=reason,
                subject=branch.subject,
                head_sha=branch.head_sha,
                unique_commit_count=branch.unique_commit_count,
                upstream=branch.upstream,
                committed_at=branch.committed_at.isoformat(),
                worktree_paths=[worktree.path for worktree in attached],
            )
        )

    return decisions


def _ensure_gh_auth(repo_root: Path) -> None:
    proc = _run(["gh", "auth", "status"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh auth failed")


def _push_branch(repo_root: Path, branch: str, upstream: str | None) -> None:
    args = ["git", "push"]
    if upstream:
        args.extend(["origin", branch])
    else:
        args.extend(["-u", "origin", branch])
    proc = _run(args, cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"failed to push {branch}")


def _existing_pr_number(repo_root: Path, repo: str, branch: str, base: str) -> int | None:
    proc = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--head",
            branch,
            "--base",
            base,
            "--state",
            "open",
            "--json",
            "number",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip() or proc.stdout.strip() or "failed to query existing PR"
        )
    payload = json.loads(proc.stdout or "[]")
    if not payload:
        return None
    number = payload[0].get("number")
    return int(number) if isinstance(number, int) else None


def _create_pr(repo_root: Path, repo: str, branch: str, base: str) -> int:
    proc = _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            base,
            "--head",
            branch,
            "--fill",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip() or proc.stdout.strip() or f"failed to create PR for {branch}"
        )
    number = _existing_pr_number(repo_root, repo, branch, base)
    if number is None:
        raise RuntimeError(f"created PR for {branch} but could not determine PR number")
    return number


def _add_labels(repo_root: Path, repo: str, number: int, labels: list[str]) -> None:
    if not labels:
        return
    proc = _run(
        ["gh", "pr", "edit", str(number), "--repo", repo, "--add-label", ",".join(labels)],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        # Label availability varies by repo policy; treat this as non-fatal.
        return


def _publish_decisions(
    repo_root: Path,
    repo: str,
    base: str,
    decisions: list[PublishDecision],
    *,
    limit: int,
    labels: list[str],
) -> list[dict[str, Any]]:
    _ensure_gh_auth(repo_root)
    results: list[dict[str, Any]] = []
    published = 0

    for decision in decisions:
        if not decision.eligible:
            continue
        if published >= limit:
            break

        _push_branch(repo_root, decision.branch, decision.upstream)
        number = _existing_pr_number(repo_root, repo, decision.branch, base)
        if number is None:
            number = _create_pr(repo_root, repo, decision.branch, base)
        _add_labels(repo_root, repo, number, labels)
        published += 1
        results.append(
            {
                "branch": decision.branch,
                "pr_number": number,
                "subject": decision.subject,
                "head_sha": decision.head_sha,
            }
        )

    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish recent committed codex automation branches into GitHub PRs."
    )
    parser.add_argument("--repo", default=".", help="Path inside the target repository")
    parser.add_argument("--base", default="main", help="Base branch to compare/publish against")
    parser.add_argument(
        "--github-repo",
        default="synaptent/aragora",
        help="GitHub repository slug for gh PR operations",
    )
    parser.add_argument(
        "--since-hours",
        type=int,
        default=DEFAULT_SINCE_HOURS,
        help="Only consider branches committed within this many hours",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of eligible branches to publish in one apply run",
    )
    parser.add_argument(
        "--branch",
        action="append",
        dest="branches",
        default=[],
        help="Restrict evaluation to one or more explicit branch names",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=["codex", "codex-automation"],
        help="PR label to add after creation; may be passed multiple times",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable output",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Push eligible branches and open PRs; default is dry-run planning only",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = _repo_root(Path(args.repo))
    cutoff = datetime.now(UTC) - timedelta(hours=args.since_hours)

    branches = _local_codex_branches(repo_root)
    if args.branches:
        requested = set(args.branches)
        branches = [branch for branch in branches if branch.branch in requested]

    worktrees = _list_worktrees(repo_root)
    merged_lookup = {
        branch.branch: _branch_is_merged(repo_root, args.base, branch.branch) for branch in branches
    }
    hydrated_branches = [
        BranchSnapshot(
            branch=branch.branch,
            upstream=branch.upstream,
            head_sha=branch.head_sha,
            committed_at=branch.committed_at,
            subject=branch.subject or _branch_subject(repo_root, branch.branch),
            unique_commit_count=_branch_unique_commit_count(repo_root, args.base, branch.branch),
        )
        for branch in branches
    ]

    open_pr_heads = _open_pr_heads(repo_root, args.github_repo)
    historical_pr_branches = _branches_with_pr_history(
        repo_root,
        args.github_repo,
        [branch.branch for branch in hydrated_branches if branch.branch not in open_pr_heads],
    )
    decisions = select_publishable_branches(
        hydrated_branches,
        worktrees,
        open_pr_heads,
        cutoff=cutoff,
        base=args.base,
        is_merged=merged_lookup,
        historical_pr_branches=historical_pr_branches,
    )

    payload: dict[str, Any] = {
        "repo": str(repo_root),
        "base": args.base,
        "cutoff": cutoff.isoformat(),
        "decisions": [asdict(decision) for decision in decisions],
    }

    if args.apply:
        published = _publish_decisions(
            repo_root,
            args.github_repo,
            args.base,
            decisions,
            limit=args.limit,
            labels=list(dict.fromkeys(args.labels)),
        )
        payload["published"] = published

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for decision in decisions:
            marker = "publish" if decision.eligible else "skip"
            print(f"{marker}: {decision.branch} [{decision.reason}] {decision.subject}")
        if args.apply:
            published = payload.get("published", [])
            if published:
                print("\npublished:")
                for item in published:
                    print(f"  - {item['branch']} -> PR #{item['pr_number']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
