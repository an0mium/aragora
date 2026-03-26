#!/usr/bin/env python3
"""Merge clearly safe codex automation PRs from a normal user shell.

Codex desktop automations can inspect and summarize PRs from their sandboxed
runtime, but reliable merging should happen from a normal user environment
where ``gh`` auth and network access are available. This helper only auto-
merges obviously safe codex PRs and leaves anything sensitive or ambiguous for
the in-app PR shepherd or a human.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SAFE_CHECK_CONCLUSIONS = {"success", "neutral", "skipped"}
SENSITIVE_PATH_TOKENS = (
    "/auth/",
    "/billing/",
    "secret",
    "kubernetes/",
    "helm/",
    "terraform/",
    ".github/workflows/",
    "deploy",
)


@dataclass(frozen=True)
class PullRequestSnapshot:
    number: int
    title: str
    head_ref: str
    is_draft: bool
    mergeable: str
    body: str
    url: str
    changed_files: list[str]
    status_rollup: list[dict[str, Any]]


@dataclass(frozen=True)
class MergeDecision:
    number: int
    title: str
    head_ref: str
    eligible: bool
    reason: str
    url: str
    changed_files: list[str]


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


def _repo_root(path: Path) -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "not a git repository")
    return Path(proc.stdout.strip()).resolve()


def _ensure_gh_auth(repo_root: Path) -> None:
    proc = _run(["gh", "auth", "status"], cwd=repo_root)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh auth failed")


def _open_codex_prs(repo_root: Path, repo: str, limit: int) -> list[dict[str, Any]]:
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
            str(limit),
            "--json",
            "number,title,headRefName,isDraft,url",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to list open PRs")
    payload = json.loads(proc.stdout or "[]")
    return [
        item
        for item in payload
        if isinstance(item, dict) and str(item.get("headRefName", "")).startswith("codex/")
    ]


def _pr_metadata(repo_root: Path, repo: str, number: int) -> dict[str, Any]:
    proc = _run(
        [
            "gh",
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "mergeable,body,statusCheckRollup",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"failed to inspect PR #{number}")
    payload = json.loads(proc.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"unexpected PR payload for #{number}")
    return payload


def _pr_changed_files(repo_root: Path, repo: str, number: int) -> list[str]:
    proc = _run(
        ["gh", "pr", "diff", str(number), "--repo", repo, "--name-only"],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"failed to inspect files for PR #{number}")
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def collect_pull_requests(repo_root: Path, repo: str, *, limit: int) -> list[PullRequestSnapshot]:
    snapshots: list[PullRequestSnapshot] = []
    for pr in _open_codex_prs(repo_root, repo, limit):
        number = int(pr["number"])
        metadata = _pr_metadata(repo_root, repo, number)
        snapshots.append(
            PullRequestSnapshot(
                number=number,
                title=str(pr.get("title", "")),
                head_ref=str(pr.get("headRefName", "")),
                is_draft=bool(pr.get("isDraft", False)),
                mergeable=str(metadata.get("mergeable", "")),
                body=str(metadata.get("body", "") or ""),
                url=str(pr.get("url", "")),
                changed_files=_pr_changed_files(repo_root, repo, number),
                status_rollup=list(metadata.get("statusCheckRollup") or []),
            )
        )
    return snapshots


def _status_reason(items: list[dict[str, Any]]) -> str:
    if not items:
        return "no_status_checks"
    for item in items:
        status = str(item.get("status", "") or "").lower()
        conclusion = str(item.get("conclusion", "") or "").lower()
        if status != "completed":
            return "checks_pending"
        if conclusion not in SAFE_CHECK_CONCLUSIONS:
            return "checks_failed"
    return "checks_clear"


def _path_is_sensitive(path: str) -> bool:
    lowered = f"/{path.lower().lstrip('/')}"
    return any(token in lowered for token in SENSITIVE_PATH_TOKENS)


def _has_validation_evidence(body: str) -> bool:
    lowered = body.lower()
    return "validation" in lowered or "validated" in lowered


def select_mergeable_prs(
    pull_requests: list[PullRequestSnapshot],
    *,
    max_files: int = 6,
) -> list[MergeDecision]:
    decisions: list[MergeDecision] = []

    for pr in sorted(pull_requests, key=lambda item: item.number):
        reason = "eligible"
        if not pr.head_ref.startswith("codex/"):
            reason = "not_codex_branch"
        elif pr.is_draft:
            reason = "draft"
        elif pr.mergeable != "MERGEABLE":
            reason = "not_mergeable"
        else:
            checks_reason = _status_reason(pr.status_rollup)
            if checks_reason != "checks_clear":
                reason = checks_reason
            elif not pr.changed_files:
                reason = "no_changed_files"
            elif len(pr.changed_files) > max_files:
                reason = "too_many_files"
            elif any(_path_is_sensitive(path) for path in pr.changed_files):
                reason = "sensitive_paths"
            elif not _has_validation_evidence(pr.body):
                reason = "missing_validation"

        decisions.append(
            MergeDecision(
                number=pr.number,
                title=pr.title,
                head_ref=pr.head_ref,
                eligible=reason == "eligible",
                reason=reason,
                url=pr.url,
                changed_files=pr.changed_files,
            )
        )

    return decisions


def _merge_pr(repo_root: Path, repo: str, number: int) -> None:
    proc = _run(
        [
            "gh",
            "pr",
            "merge",
            str(number),
            "--repo",
            repo,
            "--squash",
            "--admin",
            "--delete-branch=false",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip() or proc.stdout.strip() or f"failed to merge #{number}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge clearly safe codex automation PRs from a normal shell."
    )
    parser.add_argument(
        "--repo",
        default="synaptent/aragora",
        help="GitHub repository in OWNER/NAME form (default: %(default)s)",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Path inside the repository to inspect (default: current directory)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of open codex PRs to inspect and merge (default: %(default)s)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=6,
        help="Maximum changed files allowed for auto-merge (default: %(default)s)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually merge eligible PRs instead of only reporting decisions",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON decisions and actions",
    )
    args = parser.parse_args(argv)

    repo_root = _repo_root(Path(args.root).resolve())
    _ensure_gh_auth(repo_root)

    snapshots = collect_pull_requests(repo_root, args.repo, limit=args.limit)
    decisions = select_mergeable_prs(snapshots, max_files=args.max_files)
    merged_numbers: list[int] = []

    if args.apply:
        for decision in decisions:
            if not decision.eligible:
                continue
            _merge_pr(repo_root, args.repo, decision.number)
            merged_numbers.append(decision.number)

    if args.json:
        print(
            json.dumps(
                {
                    "repo": args.repo,
                    "inspected": len(decisions),
                    "merged": merged_numbers,
                    "decisions": [asdict(decision) for decision in decisions],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for decision in decisions:
            status = "MERGED" if decision.number in merged_numbers else decision.reason
            print(f"#{decision.number} {decision.head_ref} {status} {decision.title}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI guard
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
