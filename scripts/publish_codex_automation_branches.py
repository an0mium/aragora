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
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.github_cli_health import check_github_cli_health

UTC = timezone.utc
DEFAULT_SINCE_HOURS = 72
DEFAULT_PUBLISH_LIMIT = 2
DEFAULT_MAX_OPEN_PRS = 12
DEFAULT_COMMAND_TIMEOUT_SECONDS = 45
DEFAULT_GIT_TIMEOUT_SECONDS = 60
DEFAULT_SCAN_LIMIT = 12
CODEX_BRANCH_PREFIX = "codex/"
DEFAULT_PREFLIGHT_SCRIPT = "scripts/automation_pr_preflight.sh"
DEFAULT_PRE_PUSH_SKIP_HOOKS = "mypy-baseline"
DEFAULT_OUTBOX_DIR = Path(".aragora/automation-outbox")
VERIFY_AUTOMATION_GIT_PUSH_ENV = "ARAGORA_AUTOMATION_GIT_PUSH_VERIFY"
UNHEALTHY_OPEN_PR_MERGE_STATES = {"BLOCKED", "DIRTY"}
UNHEALTHY_CHECK_STATES = {
    "ACTION_REQUIRED",
    "ERROR",
    "FAILURE",
    "FAILED",
    "TIMED_OUT",
}
HEALTHY_CHECK_STATES = {
    "SUCCESS",
    "NEUTRAL",
    "SKIPPED",
}
CANCELLED_ADVISORY_WORKFLOWS = {
    "Metrics Drift",
    "Module Tier Drift",
    "PR Admission Controller",
}
STOPWORDS = {
    "and",
    "are",
    "automation",
    "autonomy",
    "branches",
    "chore",
    "codex",
    "covered",
    "docs",
    "feat",
    "for",
    "from",
    "fix",
    "parse",
    "publish",
    "restore",
    "resolved",
    "skip",
    "test",
    "the",
    "to",
    "with",
    "work",
}
ACTIVE_SESSION_FILES = (
    ".claude-session-active",
    ".codex_session_active",
    ".nomic-session-active",
)

try:
    from aragora.swarm.github_app_auth import gh_subprocess_run, github_cli_env
except Exception:  # pragma: no cover - fallback for partially bootstrapped script contexts

    def github_cli_env(
        base_env: Mapping[str, str] | None = None,
        *,
        prefer_app: bool = True,
    ) -> dict[str, str]:
        return dict(os.environ if base_env is None else base_env)

    def gh_subprocess_run(
        args: Sequence[str],
        *,
        timeout: float = 30.0,
        prefer_app: bool = True,
        write_op: bool = False,
        env: Mapping[str, str] | None = None,
        max_retries: int = 3,
        base_backoff: float = 5.0,
        max_backoff: float = 600.0,
        sleep: Callable[[float], None] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del prefer_app, write_op, max_retries, base_backoff, max_backoff, sleep
        return subprocess.run(
            ["gh", *list(args)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ if env is None else env),
            check=False,
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


def _gh_write_op(args: list[str]) -> bool:
    return len(args) >= 2 and (args[0], args[1]) in {
        ("pr", "create"),
        ("pr", "edit"),
    }


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = github_cli_env(os.environ) if args and args[0] == "gh" else None
    if env_overrides:
        env = dict(os.environ if env is None else env)
        env.update(env_overrides)
    if args and args[0] == "gh":
        timeout = int(
            os.environ.get(
                "ARAGORA_AUTOMATION_GH_TIMEOUT_SECONDS",
                str(DEFAULT_COMMAND_TIMEOUT_SECONDS),
            )
        )
        return gh_subprocess_run(
            args[1:],
            timeout=timeout,
            prefer_app=True,
            write_op=_gh_write_op(args[1:]),
            env=dict(os.environ if env is None else env),
            max_retries=0,
        )
    else:
        timeout = int(
            os.environ.get(
                "ARAGORA_AUTOMATION_GIT_TIMEOUT_SECONDS",
                str(DEFAULT_GIT_TIMEOUT_SECONDS),
            )
        )
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=check,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        message = stderr or f"command timed out after {timeout}s: {' '.join(args)}"
        return subprocess.CompletedProcess(args=args, returncode=124, stdout=stdout, stderr=message)


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


def _branch_patch_equivalent_to_base(repo_root: Path, base: str, branch: str) -> bool:
    proc = _run(["git", "cherry", base, branch], cwd=repo_root)
    if proc.returncode != 0:
        return False
    statuses = [line[:1] for line in proc.stdout.splitlines() if line.strip()]
    return bool(statuses) and all(status == "-" for status in statuses)


def _branch_has_pr_diff(repo_root: Path, base: str, branch: str) -> bool:
    proc = _run(["git", "diff", "--quiet", f"{base}...{branch}", "--"], cwd=repo_root)
    if proc.returncode == 0:
        return False
    if proc.returncode == 1:
        return True
    # Fail open on unexpected git errors: a missing ref or transient git
    # failure should not silently suppress a branch as "empty_pr_diff".
    return True


def _same_git_origin(left: Path, right: Path) -> bool:
    left_proc = _run(["git", "config", "--get", "remote.origin.url"], cwd=left)
    right_proc = _run(["git", "config", "--get", "remote.origin.url"], cwd=right)
    if left_proc.returncode != 0 or right_proc.returncode != 0:
        return False
    return bool(left_proc.stdout.strip()) and left_proc.stdout.strip() == right_proc.stdout.strip()


def _automation_state_root(repo_root: Path) -> Path:
    if (repo_root / ".aragora").is_dir():
        return repo_root

    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    candidates: list[tuple[Path, bool]] = []
    if configured:
        candidates.append((Path(configured).expanduser(), True))
    candidates.append((Path.home() / "Development" / "aragora", False))

    for candidate, explicit in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if not (resolved / ".aragora").is_dir():
            continue
        if explicit or _same_git_origin(repo_root, resolved):
            return resolved
    return repo_root


def _automation_state_path(repo_root: Path, path: Path | None, default_relative: Path) -> Path:
    if path is not None:
        return path if path.is_absolute() else repo_root / path
    return _automation_state_root(repo_root) / default_relative


def _json_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.glob("*.json") if item.is_file())


def _add_branch_reference(branches: set[str], value: Any) -> None:
    if isinstance(value, str):
        branch = value.strip()
        if branch.startswith(CODEX_BRANCH_PREFIX):
            branches.add(branch)


def _superseded_branches_from_payload(payload: dict[str, Any]) -> set[str]:
    branches: set[str] = set()

    def collect(container: Mapping[str, Any]) -> None:
        _add_branch_reference(branches, container.get("supersedes_branch"))
        supersedes_branches = container.get("supersedes_branches")
        if isinstance(supersedes_branches, list):
            for item in supersedes_branches:
                _add_branch_reference(branches, item)
        supersedes = container.get("supersedes")
        if isinstance(supersedes, list):
            for item in supersedes:
                _add_branch_reference(branches, item)
        else:
            _add_branch_reference(branches, supersedes)

    collect(payload)
    local_evidence = payload.get("local_evidence")
    if isinstance(local_evidence, Mapping):
        collect(local_evidence)
    elif isinstance(local_evidence, list):
        for item in local_evidence:
            if isinstance(item, Mapping):
                collect(item)
    return branches


def outbox_superseded_branches(
    repo_root: Path,
    *,
    outbox_dir: Path | None = None,
) -> set[str]:
    outbox_root = _automation_state_path(repo_root, outbox_dir, DEFAULT_OUTBOX_DIR)
    superseded: set[str] = set()
    for outbox_file in _json_files(outbox_root):
        try:
            payload = json.loads(outbox_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            superseded.update(_superseded_branches_from_payload(payload))
    return superseded


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
    # Ignore untracked files here so unrelated local docs/scratch files in an
    # attached worktree do not block publishing an already committed branch.
    proc = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path)
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def _list_worktrees(
    repo_root: Path,
    *,
    branch_filter: set[str] | None = None,
) -> list[WorktreeSnapshot]:
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
        if branch_filter is not None and current_branch not in branch_filter:
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


def _open_codex_prs(repo_root: Path, repo: str) -> list[dict[str, Any]]:
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
            "number,title,headRefName,isDraft,mergeStateStatus,reviewDecision,statusCheckRollup",
        ],
        cwd=repo_root,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "failed to list open PRs")
    payload = json.loads(proc.stdout or "[]")
    return [
        item
        for item in payload
        if isinstance(item, dict)
        and isinstance(item.get("headRefName"), str)
        and item["headRefName"].startswith(CODEX_BRANCH_PREFIX)
    ]


def _open_pr_heads(repo_root: Path, repo: str) -> set[str]:
    return {
        item["headRefName"]
        for item in _open_codex_prs(repo_root, repo)
        if isinstance(item.get("headRefName"), str)
    }


def _rollup_state(item: dict[str, Any]) -> str:
    for key in ("conclusion", "state", "status"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def _check_rollup_is_unhealthy(item: dict[str, Any]) -> bool:
    state = _rollup_state(item)
    if state in UNHEALTHY_CHECK_STATES:
        return True
    if state != "CANCELLED":
        return False

    workflow_name = str(item.get("workflowName") or "")
    return workflow_name not in CANCELLED_ADVISORY_WORKFLOWS


def _check_rollup_identity(item: dict[str, Any]) -> tuple[str, str]:
    return (str(item.get("workflowName") or ""), str(item.get("name") or ""))


def _check_rollup_label(item: dict[str, Any]) -> str:
    workflow_name, check_name = _check_rollup_identity(item)
    return "/".join(part for part in (workflow_name, check_name) if part) or "unknown-check"


def _unhealthy_check_rollup_items(check_rollup: Any) -> list[dict[str, Any]]:
    if not isinstance(check_rollup, list):
        return []

    checks = [check for check in check_rollup if isinstance(check, dict)]
    healthy_identities = {
        _check_rollup_identity(check)
        for check in checks
        if _rollup_state(check) in HEALTHY_CHECK_STATES
    }
    unhealthy: list[dict[str, Any]] = []
    for check in checks:
        if not _check_rollup_is_unhealthy(check):
            continue
        if (
            _rollup_state(check) == "CANCELLED"
            and _check_rollup_identity(check) in healthy_identities
        ):
            continue
        unhealthy.append(check)
    return unhealthy


def _open_codex_pr_unhealthy_reasons(item: dict[str, Any]) -> list[str]:
    merge_state = str(item.get("mergeStateStatus") or "UNKNOWN").upper()
    if merge_state == "DIRTY":
        return ["merge_state=DIRTY"]
    if merge_state != "BLOCKED":
        return []
    if item.get("isDraft") is True:
        return ["draft=true"]

    unhealthy_checks = _unhealthy_check_rollup_items(item.get("statusCheckRollup") or [])
    if unhealthy_checks:
        return [
            f"check={_check_rollup_label(check)}:{_rollup_state(check)}"
            for check in unhealthy_checks
        ]

    review_decision = str(item.get("reviewDecision") or "").upper()
    if review_decision == "REVIEW_REQUIRED":
        return []
    if review_decision == "CHANGES_REQUESTED":
        return ["review_decision=CHANGES_REQUESTED"]
    return [f"review_decision={review_decision or 'UNKNOWN'}"]


def _open_codex_pr_is_unhealthy(item: dict[str, Any]) -> bool:
    """Return true only for PR states that should pause more automation publishing.

    GitHub reports review-required but otherwise green PRs as
    ``mergeStateStatus=BLOCKED``. That is a human-review gate, not an unhealthy
    code queue, and should not stop the publisher while the open PR count remains
    below budget.
    """

    return bool(_open_codex_pr_unhealthy_reasons(item))


def _branch_remote_head(repo_root: Path, branch: str) -> str | None:
    proc = _run(["git", "rev-parse", f"refs/remotes/origin/{branch}"], cwd=repo_root)
    if proc.returncode != 0:
        return None
    remote_head = proc.stdout.strip()
    return remote_head or None


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


def _subject_tokens(subject: str) -> set[str]:
    return set(_ordered_subject_tokens(subject))


def _ordered_subject_tokens(subject: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]+", subject.lower()):
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        if len(token) < 3 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def _looks_related_subject(candidate: str, existing: str) -> bool:
    candidate_tokens = _subject_tokens(candidate)
    existing_tokens = _subject_tokens(existing)
    if not candidate_tokens or not existing_tokens:
        return candidate.strip().lower() == existing.strip().lower()
    overlap = candidate_tokens & existing_tokens
    return len(overlap) >= min(3, len(candidate_tokens))


def _related_search_queries(subject: str) -> list[str]:
    stable_tokens = _ordered_subject_tokens(subject)
    queries = [subject]
    if len(stable_tokens) >= 3:
        queries.append(" ".join(stable_tokens))
    return list(dict.fromkeys(query for query in queries if query.strip()))


def _branches_with_resolved_related_work(
    repo_root: Path,
    repo: str,
    branches: list[BranchSnapshot],
) -> set[str]:
    resolved: set[str] = set()
    for branch in branches:
        for query in _related_search_queries(branch.subject):
            for command in (
                [
                    "gh",
                    "pr",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "all",
                    "--search",
                    query,
                    "--json",
                    "title,state",
                    "--limit",
                    "20",
                ],
                [
                    "gh",
                    "issue",
                    "list",
                    "--repo",
                    repo,
                    "--state",
                    "all",
                    "--search",
                    query,
                    "--json",
                    "title,state",
                    "--limit",
                    "20",
                ],
            ):
                proc = _run(command, cwd=repo_root)
                if proc.returncode != 0:
                    continue
                payload = json.loads(proc.stdout or "[]")
                if not isinstance(payload, list):
                    continue
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    state = str(item.get("state") or "").upper()
                    title = str(item.get("title") or "")
                    if state in {"MERGED", "CLOSED"} and _looks_related_subject(
                        branch.subject,
                        title,
                    ):
                        resolved.add(branch.branch)
                        break
                if branch.branch in resolved:
                    break
            if branch.branch in resolved:
                break
    return resolved


def select_publishable_branches(
    branches: list[BranchSnapshot],
    worktrees: list[WorktreeSnapshot],
    open_pr_heads: set[str],
    *,
    cutoff: datetime,
    base: str,
    is_merged: dict[str, bool] | None = None,
    is_patch_equivalent: dict[str, bool] | None = None,
    historical_pr_branches: set[str] | None = None,
    resolved_related_branches: set[str] | None = None,
    remote_head_lookup: dict[str, str | None] | None = None,
    has_pr_diff: dict[str, bool] | None = None,
    superseded_outbox_branches: set[str] | None = None,
) -> list[PublishDecision]:
    worktrees_by_branch: dict[str, list[WorktreeSnapshot]] = {}
    for worktree in worktrees:
        if worktree.branch:
            worktrees_by_branch.setdefault(worktree.branch, []).append(worktree)

    merged_lookup = is_merged or {}
    patch_equivalent_lookup = is_patch_equivalent or {}
    historical_lookup = historical_pr_branches or set()
    resolved_related_lookup = resolved_related_branches or set()
    remote_lookup = remote_head_lookup or {}
    pr_diff_lookup = has_pr_diff or {}
    superseded_lookup = superseded_outbox_branches or set()
    decisions: list[PublishDecision] = []

    for branch in sorted(branches, key=lambda item: item.committed_at, reverse=True):
        attached = worktrees_by_branch.get(branch.branch, [])
        reason = "eligible"

        if merged_lookup.get(branch.branch, False):
            reason = "already_merged"
        elif patch_equivalent_lookup.get(branch.branch, False):
            reason = "patch_equivalent_to_base"
        elif branch.branch in superseded_lookup:
            reason = "superseded_by_outbox_handoff"
        elif branch.branch in resolved_related_lookup:
            reason = "related_resolved_work_exists"
        elif branch.unique_commit_count <= 0:
            reason = "no_unique_commits"
        elif pr_diff_lookup.get(branch.branch, True) is False:
            reason = "empty_pr_diff"
        elif branch.committed_at < cutoff:
            reason = "older_than_cutoff"
        elif branch.branch in open_pr_heads:
            reason = "open_pr_exists"
        elif branch.branch in historical_lookup:
            reason = "historical_pr_exists"
        elif remote_lookup.get(branch.branch) not in (None, "", branch.head_sha):
            reason = "remote_branch_conflict"
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
    health = check_github_cli_health(repo_root)
    if not health.ready:
        raise RuntimeError(health.error or health.mode)


def _github_base_ref(base: str) -> str:
    return base.removeprefix("origin/")


def _merge_skip_hooks(existing: str | None, additions: str) -> str:
    merged: list[str] = []
    for raw in (existing or "").split(",") + additions.split(","):
        hook_id = raw.strip()
        if hook_id and hook_id not in merged:
            merged.append(hook_id)
    return ",".join(merged)


def _push_branch(repo_root: Path, branch: str, upstream: str | None) -> None:
    args = ["git", "push"]
    verify_push = os.environ.get(VERIFY_AUTOMATION_GIT_PUSH_ENV, "").strip().lower()
    if verify_push not in {"1", "true", "yes"}:
        args.append("--no-verify")
    if upstream:
        args.extend(["origin", branch])
    else:
        args.extend(["-u", "origin", branch])
    pre_push_skip = os.environ.get(
        "ARAGORA_AUTOMATION_PRE_PUSH_SKIP",
        DEFAULT_PRE_PUSH_SKIP_HOOKS,
    ).strip()
    env_overrides = None
    if pre_push_skip:
        env_overrides = {"SKIP": _merge_skip_hooks(os.environ.get("SKIP"), pre_push_skip)}
    proc = _run(args, cwd=repo_root, env_overrides=env_overrides)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"failed to push {branch}")


def _existing_pr_number(repo_root: Path, repo: str, branch: str, base: str) -> int | None:
    github_base = _github_base_ref(base)
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
            github_base,
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
    github_base = _github_base_ref(base)
    proc = _run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            github_base,
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


def _run_publish_preflight(
    repo_root: Path,
    base: str,
    branch: str,
    preflight_script: str,
) -> tuple[bool, str]:
    script_path = repo_root / preflight_script
    if not script_path.exists():
        return False, f"preflight script not found: {preflight_script}"

    proc = _run(["bash", str(script_path), base, branch], cwd=repo_root)
    output = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip()).strip()
    if proc.returncode == 0:
        return True, output
    return False, output or f"preflight failed for {branch}"


def _publish_decisions(
    repo_root: Path,
    repo: str,
    base: str,
    decisions: list[PublishDecision],
    *,
    limit: int,
    open_pr_count: int,
    max_open_prs: int,
    labels: list[str],
    preflight_script: str | None = None,
) -> list[dict[str, Any]]:
    _ensure_gh_auth(repo_root)
    results: list[dict[str, Any]] = []
    published = 0

    for decision in decisions:
        if not decision.eligible:
            continue
        if published >= limit:
            break
        if open_pr_count + published >= max_open_prs:
            break

        if preflight_script:
            ok, output = _run_publish_preflight(
                repo_root,
                base,
                decision.branch,
                preflight_script,
            )
            if not ok:
                results.append(
                    {
                        "branch": decision.branch,
                        "status": "preflight_failed",
                        "subject": decision.subject,
                        "head_sha": decision.head_sha,
                        "reason": output[:2000],
                    }
                )
                continue

        try:
            _push_branch(repo_root, decision.branch, decision.upstream)
            number = _existing_pr_number(repo_root, repo, decision.branch, base)
            if number is None:
                number = _create_pr(repo_root, repo, decision.branch, base)
            _add_labels(repo_root, repo, number, labels)
        except RuntimeError as exc:
            results.append(
                {
                    "branch": decision.branch,
                    "status": "publish_failed",
                    "subject": decision.subject,
                    "head_sha": decision.head_sha,
                    "reason": str(exc)[:2000],
                }
            )
            continue
        published += 1
        results.append(
            {
                "branch": decision.branch,
                "status": "published",
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
        default=DEFAULT_PUBLISH_LIMIT,
        help="Maximum number of eligible branches to publish in one apply run",
    )
    parser.add_argument(
        "--max-open-prs",
        type=int,
        default=DEFAULT_MAX_OPEN_PRS,
        help="Maximum number of open codex PRs allowed before publishing pauses",
    )
    parser.add_argument(
        "--branch",
        action="append",
        dest="branches",
        default=[],
        help="Restrict evaluation to one or more explicit branch names",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=DEFAULT_SCAN_LIMIT,
        help="Maximum recent local codex branches to inspect when --branch is not provided.",
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
    parser.add_argument(
        "--preflight-script",
        default=DEFAULT_PREFLIGHT_SCRIPT,
        help=(
            "Repo-relative script to run before publishing each eligible branch "
            f"(default: {DEFAULT_PREFLIGHT_SCRIPT})"
        ),
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the automation PR preflight before publishing.",
    )
    parser.add_argument(
        "--allow-unhealthy-queue-publish",
        action="store_true",
        help=(
            "Publish otherwise eligible, preflighted branches even when every existing "
            "open codex PR is unhealthy. Still respects --limit and --max-open-prs."
        ),
    )
    parser.add_argument(
        "--outbox-dir",
        type=Path,
        default=None,
        help=(
            "Automation outbox directory used to skip locally superseded branches. "
            "Defaults to the shared .aragora automation state root when available."
        ),
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=None,
        help=(
            "Accepted for shared automation CLI compatibility. Branch publishing does "
            "not consume receipts directly; receipt-aware filtering lives in the "
            "backlog audit and handoff publisher helpers."
        ),
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
    else:
        branches = sorted(
            [branch for branch in branches if branch.committed_at >= cutoff],
            key=lambda branch: branch.committed_at,
            reverse=True,
        )[: max(args.scan_limit, 0)]

    branch_names = {branch.branch for branch in branches}
    worktrees = _list_worktrees(repo_root, branch_filter=branch_names)
    merged_lookup = {
        branch.branch: _branch_is_merged(repo_root, args.base, branch.branch) for branch in branches
    }
    patch_equivalent_lookup = {
        branch.branch: _branch_patch_equivalent_to_base(repo_root, args.base, branch.branch)
        for branch in branches
        if not merged_lookup.get(branch.branch, False)
    }
    pr_diff_lookup = {
        branch.branch: _branch_has_pr_diff(repo_root, args.base, branch.branch)
        for branch in branches
        if not merged_lookup.get(branch.branch, False)
    }
    superseded_outbox_lookup = outbox_superseded_branches(repo_root, outbox_dir=args.outbox_dir)
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

    github_health = check_github_cli_health(repo_root)
    if not github_health.ready:
        unavailable_payload: dict[str, Any] = {
            "repo": str(repo_root),
            "base": args.base,
            "cutoff": cutoff.isoformat(),
            "receipt_dir": str(args.receipt_dir) if args.receipt_dir else None,
            "scanned_branch_count": len(hydrated_branches),
            "open_pr_count": 0,
            "max_open_prs": args.max_open_prs,
            "github_health": github_health.to_dict(),
            "decisions": [],
        }
        if args.json:
            print(json.dumps(unavailable_payload, indent=2))
        else:
            print(f"github_unavailable: {github_health.mode} {github_health.error}".strip())
        return 0 if not hydrated_branches else 1

    open_codex_prs = _open_codex_prs(repo_root, args.github_repo)
    open_pr_heads = {
        item["headRefName"] for item in open_codex_prs if isinstance(item.get("headRefName"), str)
    }
    historical_pr_branches = _branches_with_pr_history(
        repo_root,
        args.github_repo,
        [branch.branch for branch in hydrated_branches if branch.branch not in open_pr_heads],
    )
    resolved_related_branches = _branches_with_resolved_related_work(
        repo_root,
        args.github_repo,
        [branch for branch in hydrated_branches if branch.branch not in historical_pr_branches],
    )
    decisions = select_publishable_branches(
        hydrated_branches,
        worktrees,
        open_pr_heads,
        cutoff=cutoff,
        base=args.base,
        is_merged=merged_lookup,
        is_patch_equivalent=patch_equivalent_lookup,
        historical_pr_branches=historical_pr_branches,
        resolved_related_branches=resolved_related_branches,
        remote_head_lookup={
            branch.branch: _branch_remote_head(repo_root, branch.branch)
            for branch in hydrated_branches
        },
        has_pr_diff=pr_diff_lookup,
        superseded_outbox_branches=superseded_outbox_lookup,
    )
    merge_state_counts: dict[str, int] = {}
    unhealthy_open_prs: list[dict[str, Any]] = []
    for item in open_codex_prs:
        state = str(item.get("mergeStateStatus") or "UNKNOWN").upper()
        merge_state_counts[state] = merge_state_counts.get(state, 0) + 1
        unhealthy_reasons = _open_codex_pr_unhealthy_reasons(item)
        if unhealthy_reasons:
            unhealthy_open_prs.append(
                {
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "headRefName": item.get("headRefName"),
                    "mergeStateStatus": item.get("mergeStateStatus"),
                    "reviewDecision": item.get("reviewDecision"),
                    "reasons": unhealthy_reasons,
                }
            )
    unhealthy_open_pr_count = len(unhealthy_open_prs)
    all_open_prs_unhealthy = bool(open_codex_prs) and unhealthy_open_pr_count == len(open_codex_prs)

    payload: dict[str, Any] = {
        "repo": str(repo_root),
        "base": args.base,
        "cutoff": cutoff.isoformat(),
        "receipt_dir": str(args.receipt_dir) if args.receipt_dir else None,
        "open_pr_count": len(open_pr_heads),
        "max_open_prs": args.max_open_prs,
        "queue_health": {
            "open_codex_pr_count": len(open_codex_prs),
            "unhealthy_open_pr_count": unhealthy_open_pr_count,
            "merge_state_counts": merge_state_counts,
            "all_open_prs_unhealthy": all_open_prs_unhealthy,
            "unhealthy_open_prs": unhealthy_open_prs,
        },
        "github_health": github_health.to_dict(),
        "decisions": [asdict(decision) for decision in decisions],
    }

    if args.apply:
        if all_open_prs_unhealthy and not args.allow_unhealthy_queue_publish:
            payload["published"] = []
            payload["publish_paused_reason"] = "open_pr_queue_unhealthy"
        else:
            if all_open_prs_unhealthy and args.allow_unhealthy_queue_publish:
                payload["publish_override_reason"] = "allow_unhealthy_queue_publish"
            published = _publish_decisions(
                repo_root,
                args.github_repo,
                args.base,
                decisions,
                limit=args.limit,
                open_pr_count=len(open_pr_heads),
                max_open_prs=args.max_open_prs,
                labels=list(dict.fromkeys(args.labels)),
                preflight_script=None if args.skip_preflight else str(args.preflight_script),
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
                    if item.get("status") == "preflight_failed":
                        print(f"  - {item['branch']} skipped: preflight_failed")
                    else:
                        print(f"  - {item['branch']} -> PR #{item['pr_number']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
