#!/usr/bin/env python3
"""Classify local codex/* branch backlog without deleting anything.

The automation publisher intentionally scans only a recent window. This audit is
for the separate hygiene problem: large historical local codex/* branch counts
can make automations skip useful work even when most branches are already
merged, patch-equivalent, or protected by open PR/worktree state.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.github_cli_health import check_github_cli_health

ACTIVE_SESSION_FILES = (
    ".claude-session-active",
    ".codex_session_active",
    ".nomic-session-active",
)
DEFAULT_OUTBOX_DIR = Path(".aragora/automation-outbox")
DEFAULT_RECEIPT_DIR = Path(".aragora/automation-receipts")
TERMINAL_RECEIPT_STATUSES = {"published", "already_satisfied", "completed", "skipped"}
COMMIT_PREFIX_RE = re.compile(r"^[0-9a-f]{7,40}$", re.IGNORECASE)
BRANCH_IDEMPOTENCY_PREFIXES = ("open-pr-", "already-satisfied-")
SALVAGE_CATEGORIES = {
    "salvage_recent_unique",
    "salvage_stale_remote_unique",
    "salvage_stale_local_unique",
    "salvage_diverged_recent",
    "salvage_diverged_remote",
    "salvage_diverged_local",
}
COMPACT_RECORD_EXAMPLE_FIELDS = (
    "name",
    "category",
    "head_sha",
    "committed_at",
    "subject",
    "ahead_count",
    "behind_count",
    "open_pr",
    "worktree_paths",
    "active_worktree_paths",
    "dirty_worktree_paths",
    "handoff_receipt_exists",
    "handoff_outbox_exists",
)
DEFAULT_SUMMARY_EXAMPLES_PER_CATEGORY = 3


@dataclass(frozen=True)
class BranchRecord:
    name: str
    upstream: str | None
    head_sha: str
    committed_at: str
    subject: str
    ahead_count: int
    behind_count: int
    merged_to_base: bool
    patch_equivalent_to_base: bool
    remote_branch_exists: bool
    open_pr: int | None
    worktree_paths: list[str]
    dirty_worktree_paths: list[str]
    active_worktree_paths: list[str]
    handoff_receipt_exists: bool
    handoff_outbox_exists: bool
    category: str


def run_git(args: list[str], cwd: Path, *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    cmd = ["git", *args]
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        message = stderr or f"command timed out after {timeout}s: {' '.join(cmd)}"
        return subprocess.CompletedProcess(args=cmd, returncode=124, stdout=stdout, stderr=message)


def run_gh(args: list[str], cwd: Path, *, timeout: int = 45) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def repo_root(path: Path) -> Path:
    proc = run_git(["rev-parse", "--show-toplevel"], path)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "not a git repository")
    return Path(proc.stdout.strip()).resolve()


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def local_branches(root: Path, prefix: str, base: str) -> list[dict[str, str]]:
    ref_prefix = f"refs/heads/{prefix}"
    proc = run_git(
        [
            "for-each-ref",
            "--format=%(refname:short)|%(upstream:short)|%(objectname:short)|%(committerdate:iso8601)|%(subject)",
            ref_prefix,
        ],
        root,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "failed to enumerate local branches")

    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        name, upstream, head_sha, committed_at, subject = line.split("|", 4)
        rows.append(
            {
                "name": name,
                "upstream": upstream,
                "head_sha": head_sha,
                "committed_at": committed_at,
                "ahead_count": "",
                "behind_count": "",
                "subject": subject,
            }
        )
    return rows


def remote_branch_names(root: Path, prefix: str) -> set[str]:
    proc = run_git(
        ["for-each-ref", "--format=%(refname:short)", f"refs/remotes/origin/{prefix}"], root
    )
    if proc.returncode != 0:
        return set()
    remote_prefix = "origin/"
    return {
        line.removeprefix(remote_prefix)
        for line in proc.stdout.splitlines()
        if line.startswith(remote_prefix)
    }


def merged_branch_names(root: Path, base: str, prefix: str) -> set[str]:
    proc = run_git(["branch", "--format=%(refname:short)", "--merged", base], root)
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip().startswith(prefix)}


def worktree_map(root: Path) -> dict[str, list[Path]]:
    proc = run_git(["worktree", "list", "--porcelain"], root)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "failed to list worktrees")

    by_branch: dict[str, list[Path]] = defaultdict(list)
    current_path: Path | None = None
    current_branch: str | None = None

    def flush() -> None:
        if current_path is not None and current_branch:
            by_branch[current_branch].append(current_path)

    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            flush()
            current_path = Path(line.removeprefix("worktree ").strip()).resolve()
            current_branch = None
        elif line.startswith("branch "):
            current_branch = line.removeprefix("branch refs/heads/").strip()
    flush()
    return by_branch


def dirty_worktree(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        proc = run_git(["status", "--porcelain"], path)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return proc.returncode == 0 and bool(proc.stdout.strip())


def active_worktree(path: Path) -> bool:
    return any((path / marker).exists() for marker in ACTIVE_SESSION_FILES)


def _repo_relative(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _same_git_origin(left: Path, right: Path) -> bool:
    left_proc = run_git(["config", "--get", "remote.origin.url"], left)
    right_proc = run_git(["config", "--get", "remote.origin.url"], right)
    if left_proc.returncode != 0 or right_proc.returncode != 0:
        return False
    return bool(left_proc.stdout.strip()) and left_proc.stdout.strip() == right_proc.stdout.strip()


def _automation_state_root(root: Path) -> Path:
    """Return the repo path whose .aragora state should back automation audits."""

    if (root / ".aragora").is_dir():
        return root

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
        if resolved.name == ".aragora" and resolved.is_dir():
            return resolved
        if not (resolved / ".aragora").is_dir():
            continue
        if explicit or _same_git_origin(root, resolved):
            return resolved
    return root


def _automation_state_path(root: Path, path: Path | None, default_relative: Path) -> Path:
    if path is not None:
        return _repo_relative(root, path)
    return _automation_state_default_path(_automation_state_root(root), default_relative)


def _automation_state_default_path(state_root: Path, default_relative: Path) -> Path:
    expanded = state_root.expanduser()
    if default_relative.parts[:1] == (".aragora",) and expanded.name == ".aragora":
        return expanded.joinpath(*default_relative.parts[1:])
    return expanded / default_relative


def _json_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.glob("*.json") if item.is_file())


def _json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def terminal_handoff_keys(receipt_root: Path) -> set[str]:
    """Return terminal automation handoff idempotency keys."""

    terminal_keys: set[str] = set()
    for receipt_file in _json_files(receipt_root):
        payload = _json_mapping(receipt_file)
        if payload is None:
            continue
        if str(payload.get("status") or "") not in TERMINAL_RECEIPT_STATUSES:
            continue
        idempotency_key = str(payload.get("idempotency_key") or receipt_file.stem).strip()
        if idempotency_key:
            terminal_keys.add(idempotency_key)
    return terminal_keys


def terminal_key_matches_branch_head(idempotency_key: str, branch: str, head_sha: str) -> bool:
    """Return whether a terminal idempotency key names this branch head."""

    if not branch or not head_sha:
        return False

    normalized_key = idempotency_key.strip().lower()
    normalized_branch = branch.replace("/", "-").lower()
    normalized_head = head_sha.strip().lower()
    for key_prefix in BRANCH_IDEMPOTENCY_PREFIXES:
        branch_prefix = f"{key_prefix}{normalized_branch}-"
        if not normalized_key.startswith(branch_prefix):
            continue
        commit_prefix = normalized_key.removeprefix(branch_prefix)
        if not COMMIT_PREFIX_RE.fullmatch(commit_prefix):
            return False
        return normalized_head.startswith(commit_prefix) or commit_prefix.startswith(
            normalized_head
        )
    return False


def _local_evidence_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _outbox_payload_branch(payload: dict[str, Any]) -> str:
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        branch = str(local_evidence.get("branch") or "").strip()
        if branch:
            return branch
    return str(payload.get("branch") or "").strip()


def _add_branch_reference(branches: set[str], value: Any) -> None:
    if isinstance(value, str):
        branch = value.strip()
        if branch:
            branches.add(branch)
    elif isinstance(value, Mapping):
        branch = str(value.get("branch") or "").strip()
        if branch:
            branches.add(branch)


def _structured_action(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                return parsed
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
            if isinstance(parsed, Mapping):
                return parsed
    return None


def _outbox_payload_branches(payload: dict[str, Any]) -> set[str]:
    """Return primary and explicitly superseded branch references from a handoff."""

    branches: set[str] = set()
    _add_branch_reference(branches, _outbox_payload_branch(payload))
    requested_action = _structured_action(payload.get("requested_action"))
    if requested_action is not None:
        _add_branch_reference(branches, requested_action.get("branch"))

    containers: list[Mapping[str, Any]] = [
        *_local_evidence_mappings(payload.get("local_evidence")),
        payload,
    ]

    for container in containers:
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

    return branches


def terminal_receipt_branches(receipt_root: Path) -> set[str]:
    """Return branch references stored directly in terminal receipt payloads."""

    branches: set[str] = set()
    for receipt_file in _json_files(receipt_root):
        payload = _json_mapping(receipt_file)
        if payload is None:
            continue
        if str(payload.get("status") or "") not in TERMINAL_RECEIPT_STATUSES:
            continue
        branches.update(_outbox_payload_branches(payload))
    return branches


def _archived_outbox_receipt_branches(
    outbox_root: Path,
    receipt_root: Path,
    terminal_keys: set[str],
) -> set[str]:
    """Return branch references from archived outbox payloads named by receipts."""

    archive_root = outbox_root.parent / "automation-outbox-archive"
    if not archive_root.exists():
        return set()

    branches: set[str] = set()
    for receipt_file in _json_files(receipt_root):
        receipt = _json_mapping(receipt_file)
        if receipt is None:
            continue
        if str(receipt.get("status") or "") not in TERMINAL_RECEIPT_STATUSES:
            continue
        idempotency_key = str(receipt.get("idempotency_key") or receipt_file.stem).strip()
        if idempotency_key not in terminal_keys:
            continue

        candidates: list[Path] = []
        source_file = str(receipt.get("source_file") or "").strip()
        if source_file:
            candidates.append(archive_root / Path(source_file).name)
        if idempotency_key:
            candidates.append(archive_root / f"{idempotency_key}.json")

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            payload = _json_mapping(candidate)
            if payload is None:
                continue
            payload_key = str(payload.get("idempotency_key") or "").strip()
            if payload_key and payload_key != idempotency_key:
                continue
            branches.update(_outbox_payload_branches(payload))
    return branches


def terminal_receipted_handoff_branches(
    root: Path,
    *,
    outbox_dir: Path | None = None,
    receipt_dir: Path | None = None,
) -> set[str]:
    """Return branch names that already have terminal automation handoff receipts."""

    outbox_root = _automation_state_path(root, outbox_dir, DEFAULT_OUTBOX_DIR)
    receipt_root = _automation_state_path(root, receipt_dir, DEFAULT_RECEIPT_DIR)
    terminal_keys = terminal_handoff_keys(receipt_root)
    if not terminal_keys:
        return set()

    branches = terminal_receipt_branches(receipt_root)
    for outbox_file in _json_files(outbox_root):
        try:
            payload = json.loads(outbox_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if idempotency_key not in terminal_keys:
            continue
        branches.update(_outbox_payload_branches(payload))
    branches.update(_archived_outbox_receipt_branches(outbox_root, receipt_root, terminal_keys))
    return branches


def unresolved_outbox_handoff_branches(
    root: Path,
    *,
    outbox_dir: Path | None = None,
    receipt_dir: Path | None = None,
) -> set[str]:
    """Return branch names that already have unresolved automation outbox handoffs."""

    outbox_root = _automation_state_path(root, outbox_dir, DEFAULT_OUTBOX_DIR)
    receipt_root = _automation_state_path(root, receipt_dir, DEFAULT_RECEIPT_DIR)
    terminal_keys = terminal_handoff_keys(receipt_root)
    branches: set[str] = set()
    for outbox_file in _json_files(outbox_root):
        try:
            payload = json.loads(outbox_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not idempotency_key or idempotency_key in terminal_keys:
            continue
        branches.update(_outbox_payload_branches(payload))
    return branches


def open_pr_heads(root: Path, repo: str, prefix: str) -> dict[str, int]:
    proc = run_gh(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "200",
            "--json",
            "number,headRefName",
        ],
        root,
    )
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return {}
    heads: dict[str, int] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        head = item.get("headRefName")
        number = item.get("number")
        if isinstance(head, str) and head.startswith(prefix) and isinstance(number, int):
            heads[head] = number
    return heads


def count_ahead(root: Path, base: str, branch: str) -> int:
    ahead_count, _behind_count = branch_divergence(root, base, branch)
    return ahead_count


def branch_divergence(root: Path, base: str, branch: str) -> tuple[int, int]:
    proc = run_git(["rev-list", "--left-right", "--count", f"{base}...{branch}"], root)
    if proc.returncode != 0:
        return (0, 0)
    try:
        behind_count, ahead_count = (int(part) for part in proc.stdout.split())
        return (ahead_count, behind_count)
    except ValueError:
        return (0, 0)


def is_merged(root: Path, base: str, branch: str) -> bool:
    return run_git(["merge-base", "--is-ancestor", branch, base], root).returncode == 0


def _patch_budget_deadline(time_budget_seconds: float | None) -> float | None:
    if time_budget_seconds is None:
        return None
    return monotonic() + max(0.0, time_budget_seconds)


def _patch_budget_exhausted(deadline: float | None) -> bool:
    return deadline is not None and monotonic() >= deadline


def _patch_timeout(deadline: float | None, default: int) -> int:
    if deadline is None:
        return default
    remaining = deadline - monotonic()
    if remaining <= 0:
        return 1
    return max(1, min(default, int(remaining)))


def has_empty_branch_diff(
    root: Path,
    base: str,
    branch: str,
    *,
    timeout: int = 60,
) -> bool:
    return run_git(["diff", "--quiet", f"{base}...{branch}"], root, timeout=timeout).returncode == 0


def has_empty_changed_file_diff(
    root: Path,
    base: str,
    branch: str,
    *,
    timeout: int = 60,
) -> bool:
    """Return true when the branch-touched files already match base."""

    paths_proc = run_git(["diff", "--name-only", "-z", f"{base}...{branch}"], root, timeout=timeout)
    if paths_proc.returncode != 0:
        return False
    paths = [path for path in paths_proc.stdout.split("\0") if path]
    if not paths:
        return True
    diff_proc = run_git(
        ["diff", "--quiet", f"{base}..{branch}", "--", *paths],
        root,
        timeout=timeout,
    )
    return diff_proc.returncode == 0


def is_patch_equivalent(
    root: Path,
    base: str,
    branch: str,
    *,
    timeout: int = 60,
) -> bool:
    diff_proc = run_git(["diff", "--quiet", f"{base}...{branch}"], root, timeout=timeout)
    if diff_proc.returncode == 0:
        return True
    if diff_proc.returncode != 1:
        return False

    if has_empty_changed_file_diff(root, base, branch, timeout=timeout):
        return True

    proc = run_git(["cherry", base, branch], root, timeout=timeout)
    if proc.returncode != 0:
        return False
    statuses = [line[:1] for line in proc.stdout.splitlines() if line.strip()]
    return bool(statuses) and all(status == "-" for status in statuses)


def branch_patch_id(
    root: Path,
    base: str,
    branch: str,
    *,
    timeout: int = 120,
) -> str | None:
    """Return a stable patch-id for a branch diff against base, if available."""

    diff_proc = run_git(["diff", f"{base}...{branch}"], root, timeout=timeout)
    if diff_proc.returncode != 0 or not diff_proc.stdout.strip():
        return None
    try:
        proc = subprocess.run(
            ["git", "patch-id", "--stable"],
            cwd=root,
            text=True,
            input=diff_proc.stdout,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    first_line = next((line for line in proc.stdout.splitlines() if line.strip()), "")
    return first_line.split(" ", 1)[0] or None


def branch_patch_ids(
    root: Path,
    base: str,
    branches: set[str],
    *,
    deadline: float | None = None,
) -> set[str]:
    patch_ids: set[str] = set()
    for branch in sorted(branches):
        if _patch_budget_exhausted(deadline):
            break
        patch_id = branch_patch_id(
            root,
            base,
            branch,
            timeout=_patch_timeout(deadline, 120),
        )
        if patch_id:
            patch_ids.add(patch_id)
    return patch_ids


def classify(
    *,
    open_pr: int | None,
    active_paths: list[str],
    dirty_paths: list[str],
    ahead_count: int,
    behind_count: int,
    merged_to_base: bool,
    patch_equivalent_to_base: bool,
    handoff_receipt_exists: bool,
    handoff_outbox_exists: bool,
    committed_at: datetime,
    recent_cutoff: datetime,
    remote_branch_exists: bool,
) -> str:
    if open_pr is not None:
        return "protected_open_pr"
    if active_paths:
        return "protected_active_worktree"
    if dirty_paths:
        return "protected_dirty_worktree"
    if merged_to_base or ahead_count == 0:
        return "cleanup_local_merged"
    if patch_equivalent_to_base:
        return "cleanup_patch_equivalent"
    if handoff_receipt_exists:
        return "protected_handoff_receipt"
    if handoff_outbox_exists:
        return "protected_handoff_outbox"
    if behind_count > 0:
        if committed_at >= recent_cutoff:
            return "salvage_diverged_recent"
        if remote_branch_exists:
            return "salvage_diverged_remote"
        return "salvage_diverged_local"
    if committed_at >= recent_cutoff:
        return "salvage_recent_unique"
    if remote_branch_exists:
        return "salvage_stale_remote_unique"
    return "salvage_stale_local_unique"


def audit(
    *,
    root: Path,
    base: str,
    repo: str,
    prefix: str,
    recent_hours: int,
    max_branches: int | None,
    include_patch_equivalence: bool,
    patch_equivalence_time_budget_seconds: float | None = None,
    publisher_backlog_limit: int,
    outbox_dir: Path | None = None,
    receipt_dir: Path | None = None,
) -> dict[str, Any]:
    rows = local_branches(root, prefix, base)
    rows.sort(key=lambda row: parse_dt(row["committed_at"]), reverse=True)
    if max_branches is not None:
        rows = rows[:max_branches]

    recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=recent_hours)
    remotes = remote_branch_names(root, prefix)
    merged_branches = merged_branch_names(root, base, prefix)
    worktrees = worktree_map(root)
    github_health = check_github_cli_health(root)
    prs = open_pr_heads(root, repo, prefix) if github_health.ready else {}
    resolved_outbox_dir = _automation_state_path(root, outbox_dir, DEFAULT_OUTBOX_DIR)
    resolved_receipt_dir = _automation_state_path(root, receipt_dir, DEFAULT_RECEIPT_DIR)
    handoff_receipted_branches = terminal_receipted_handoff_branches(
        root,
        outbox_dir=resolved_outbox_dir,
        receipt_dir=resolved_receipt_dir,
    )
    terminal_keys = terminal_handoff_keys(resolved_receipt_dir)
    handoff_receipted_branches.update(
        row["name"]
        for row in rows
        if any(
            terminal_key_matches_branch_head(key, row["name"], row["head_sha"])
            for key in terminal_keys
        )
    )
    handoff_outbox_branches = unresolved_outbox_handoff_branches(
        root,
        outbox_dir=resolved_outbox_dir,
        receipt_dir=resolved_receipt_dir,
    )
    patch_deadline = _patch_budget_deadline(patch_equivalence_time_budget_seconds)
    handoff_outbox_patch_ids = set()
    handoff_receipted_patch_ids = (
        branch_patch_ids(root, base, handoff_receipted_branches, deadline=patch_deadline)
        if include_patch_equivalence
        else set()
    )
    if include_patch_equivalence and not _patch_budget_exhausted(patch_deadline):
        handoff_outbox_patch_ids = branch_patch_ids(
            root,
            base,
            handoff_outbox_branches,
            deadline=patch_deadline,
        )

    records: list[BranchRecord] = []
    patch_equivalence_skipped_branches = 0
    for row in rows:
        branch = row["name"]
        committed_at = parse_dt(row["committed_at"])
        paths = worktrees.get(branch, [])
        dirty_paths = [str(path) for path in paths if dirty_worktree(path)]
        active_paths = [str(path) for path in paths if active_worktree(path)]
        try:
            ahead_count = int(row["ahead_count"])
        except ValueError:
            ahead_count, behind_count = branch_divergence(root, base, branch)
        else:
            try:
                behind_count = int(row.get("behind_count", "0"))
            except ValueError:
                _ahead_count, behind_count = branch_divergence(root, base, branch)
        merged_to_base = branch in merged_branches
        patch_equivalent = False
        patch_id = None
        if ahead_count > 0 and not merged_to_base:
            if _patch_budget_exhausted(patch_deadline):
                patch_equivalence_skipped_branches += 1
            else:
                if include_patch_equivalence:
                    patch_equivalent = is_patch_equivalent(
                        root,
                        base,
                        branch,
                        timeout=_patch_timeout(patch_deadline, 60),
                    )
                else:
                    patch_equivalent = has_empty_branch_diff(
                        root,
                        base,
                        branch,
                        timeout=_patch_timeout(patch_deadline, 60),
                    )
                if (
                    include_patch_equivalence
                    and not patch_equivalent
                    and not _patch_budget_exhausted(patch_deadline)
                ):
                    patch_id = branch_patch_id(
                        root,
                        base,
                        branch,
                        timeout=_patch_timeout(patch_deadline, 120),
                    )
        remote_exists = branch in remotes
        handoff_receipted = branch in handoff_receipted_branches or (
            patch_id is not None and patch_id in handoff_receipted_patch_ids
        )
        handoff_outbox = branch in handoff_outbox_branches or (
            patch_id is not None and patch_id in handoff_outbox_patch_ids
        )
        category = classify(
            open_pr=prs.get(branch),
            active_paths=active_paths,
            dirty_paths=dirty_paths,
            ahead_count=ahead_count,
            behind_count=behind_count,
            merged_to_base=merged_to_base,
            patch_equivalent_to_base=patch_equivalent,
            handoff_receipt_exists=handoff_receipted,
            handoff_outbox_exists=handoff_outbox,
            committed_at=committed_at,
            recent_cutoff=recent_cutoff,
            remote_branch_exists=remote_exists,
        )
        if (
            not include_patch_equivalence
            and not patch_equivalent
            and category in SALVAGE_CATEGORIES
        ):
            if _patch_budget_exhausted(patch_deadline):
                patch_equivalence_skipped_branches += 1
            else:
                patch_equivalent = is_patch_equivalent(
                    root,
                    base,
                    branch,
                    timeout=_patch_timeout(patch_deadline, 60),
                )
            if patch_equivalent:
                category = classify(
                    open_pr=prs.get(branch),
                    active_paths=active_paths,
                    dirty_paths=dirty_paths,
                    ahead_count=ahead_count,
                    behind_count=behind_count,
                    merged_to_base=merged_to_base,
                    patch_equivalent_to_base=patch_equivalent,
                    handoff_receipt_exists=handoff_receipted,
                    handoff_outbox_exists=handoff_outbox,
                    committed_at=committed_at,
                    recent_cutoff=recent_cutoff,
                    remote_branch_exists=remote_exists,
                )
        records.append(
            BranchRecord(
                name=branch,
                upstream=row["upstream"] or None,
                head_sha=row["head_sha"],
                committed_at=committed_at.isoformat(),
                subject=row["subject"],
                ahead_count=ahead_count,
                behind_count=behind_count,
                merged_to_base=merged_to_base,
                patch_equivalent_to_base=patch_equivalent,
                remote_branch_exists=remote_exists,
                open_pr=prs.get(branch),
                worktree_paths=[str(path) for path in paths],
                dirty_worktree_paths=dirty_paths,
                active_worktree_paths=active_paths,
                handoff_receipt_exists=handoff_receipted,
                handoff_outbox_exists=handoff_outbox,
                category=category,
            )
        )

    counts = Counter(record.category for record in records)
    safe_cleanup = counts["cleanup_local_merged"] + counts["cleanup_patch_equivalent"]
    protected = (
        counts["protected_open_pr"]
        + counts["protected_active_worktree"]
        + counts["protected_dirty_worktree"]
        + counts["protected_handoff_receipt"]
        + counts["protected_handoff_outbox"]
    )
    salvage = (
        counts["salvage_recent_unique"]
        + counts["salvage_stale_remote_unique"]
        + counts["salvage_stale_local_unique"]
        + counts["salvage_diverged_recent"]
        + counts["salvage_diverged_remote"]
        + counts["salvage_diverged_local"]
    )
    diverged_salvage = (
        counts["salvage_diverged_recent"]
        + counts["salvage_diverged_remote"]
        + counts["salvage_diverged_local"]
    )
    publishable_branch_backlog = (
        counts["salvage_recent_unique"] + counts["salvage_stale_remote_unique"]
    )
    return {
        "repo": str(root),
        "base": base,
        "prefix": prefix,
        "recent_hours": recent_hours,
        "publisher_backlog_limit": publisher_backlog_limit,
        "include_patch_equivalence": include_patch_equivalence,
        "patch_equivalence_time_budget_seconds": patch_equivalence_time_budget_seconds,
        "patch_equivalence_budget_exhausted": _patch_budget_exhausted(patch_deadline),
        "patch_equivalence_skipped_branches": patch_equivalence_skipped_branches,
        "outbox_dir": str(resolved_outbox_dir),
        "receipt_dir": str(resolved_receipt_dir),
        "github_health": github_health.to_dict(),
        "open_pr_lookup_skipped": not github_health.ready,
        "branch_count": len(records),
        "summary": {
            "safe_cleanup_candidates": safe_cleanup,
            "protected": protected,
            "salvage_candidates": salvage,
            "publishable_branch_backlog": publishable_branch_backlog,
            "diverged_salvage_candidates": diverged_salvage,
            "stale_local_only_salvage_candidates": counts["salvage_stale_local_unique"],
            "handoff_receipted_branches": counts["protected_handoff_receipt"],
            "handoff_outbox_branches": counts["protected_handoff_outbox"],
            "writer_should_pause_for_branch_backlog": (
                publishable_branch_backlog >= publisher_backlog_limit
            ),
            "by_category": dict(sorted(counts.items())),
        },
        "records": [asdict(record) for record in records],
    }


def compact_record_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    limit_per_category: int = DEFAULT_SUMMARY_EXAMPLES_PER_CATEGORY,
) -> dict[str, list[dict[str, Any]]]:
    """Return small per-category record examples for compact automation output."""

    if limit_per_category <= 0:
        return {}

    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        category = str(record.get("category") or "").strip()
        if not category or len(examples[category]) >= limit_per_category:
            continue
        examples[category].append(
            {field: record[field] for field in COMPACT_RECORD_EXAMPLE_FIELDS if field in record}
        )
    return dict(sorted(examples.items()))


def summary_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compact = dict(payload)
    records = payload.get("records")
    compact["record_examples"] = compact_record_examples(
        records if isinstance(records, Sequence) and not isinstance(records, (str, bytes)) else []
    )
    compact["record_examples_limit"] = DEFAULT_SUMMARY_EXAMPLES_PER_CATEGORY
    compact["records"] = []
    compact["records_omitted"] = True
    return compact


def print_markdown(payload: dict[str, Any], *, examples: int) -> None:
    summary = payload["summary"]
    print("# Codex Branch Backlog Audit\n")
    print(f"- Repo: `{payload['repo']}`")
    print(f"- Base: `{payload['base']}`")
    print(f"- Branches audited: `{payload['branch_count']}`")
    print(f"- Safe cleanup candidates: `{summary['safe_cleanup_candidates']}`")
    print(f"- Salvage candidates: `{summary['salvage_candidates']}`")
    print(f"- Publishable branch backlog: `{summary['publishable_branch_backlog']}`")
    print(f"- Diverged salvage candidates: `{summary['diverged_salvage_candidates']}`")
    print(f"- Handoff-receipted branches: `{summary['handoff_receipted_branches']}`")
    print(f"- Handoff-outbox branches: `{summary['handoff_outbox_branches']}`")
    print(
        "- Writer should pause for branch backlog: "
        f"`{summary['writer_should_pause_for_branch_backlog']}`"
    )
    print(f"- Protected branches: `{summary['protected']}`\n")
    print("## Counts\n")
    for category, count in summary["by_category"].items():
        print(f"- `{category}`: `{count}`")

    records = payload["records"]
    for category in (
        "salvage_recent_unique",
        "salvage_stale_remote_unique",
        "salvage_stale_local_unique",
        "salvage_diverged_recent",
        "salvage_diverged_remote",
        "salvage_diverged_local",
        "cleanup_local_merged",
        "cleanup_patch_equivalent",
        "protected_open_pr",
        "protected_active_worktree",
        "protected_dirty_worktree",
        "protected_handoff_receipt",
        "protected_handoff_outbox",
    ):
        matches = [record for record in records if record["category"] == category]
        if not matches:
            continue
        print(f"\n## {category}\n")
        for record in matches[:examples]:
            print(
                f"- `{record['name']}` "
                f"ahead={record['ahead_count']} "
                f"behind={record['behind_count']} "
                f"sha={record['head_sha']} "
                f"subject={record['subject']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify local codex/* branch backlog without deleting anything."
    )
    parser.add_argument("--repo", default=".", help="Path inside the repository")
    parser.add_argument("--base", default="origin/main", help="Base ref for branch comparison")
    parser.add_argument("--github-repo", default="synaptent/aragora", help="GitHub repo slug")
    parser.add_argument("--prefix", default="codex/", help="Local branch prefix to audit")
    parser.add_argument(
        "--recent-hours",
        type=int,
        default=72,
        help="Age window for recent unique salvage candidates",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs; default audits all matching branches",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument("--markdown", action="store_true", help="Print Markdown")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit detailed per-branch records from output for compact automation gating.",
    )
    parser.add_argument(
        "--include-patch-equivalence",
        action="store_true",
        default=True,
        help="Run git cherry per non-merged branch to identify patch-equivalent cleanup candidates.",
    )
    parser.add_argument(
        "--skip-patch-equivalence",
        dest="include_patch_equivalence",
        action="store_false",
        help=(
            "Skip broad git cherry patch-equivalence checks for a faster approximate "
            "audit; zero-diff cleanup and otherwise-unprotected salvage candidates "
            "are still checked."
        ),
    )
    parser.add_argument(
        "--patch-equivalence-time-budget-seconds",
        type=float,
        default=90.0,
        help=(
            "Wall-clock budget for patch-equivalence and patch-id checks. "
            "Use 0 to skip them immediately or a negative value for no budget."
        ),
    )
    parser.add_argument(
        "--publisher-backlog-limit",
        type=int,
        default=12,
        help=(
            "Threshold for publishable branch backlog. This intentionally excludes "
            "stale local-only codex/* branches so writer automations do not pause "
            "on historical local ref cache."
        ),
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help=(
            "Shared automation state root used to derive default handoff dirs. "
            "Accepts either a repo root containing .aragora or the .aragora "
            "directory itself. Explicit --outbox-dir/--receipt-dir override it."
        ),
    )
    parser.add_argument(
        "--outbox-dir",
        type=Path,
        default=None,
        help=(
            "Automation outbox directory to use for terminal handoff receipt matching. "
            "Relative paths are resolved from the repo root. Defaults to the repo "
            "state root, falling back to ARAGORA_AUTOMATION_STATE_ROOT when this "
            "worktree has no .aragora directory."
        ),
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=None,
        help=(
            "Automation receipt directory to use for terminal handoff receipt matching. "
            "Relative paths are resolved from the repo root. Defaults to the repo "
            "state root, falling back to ARAGORA_AUTOMATION_STATE_ROOT when this "
            "worktree has no .aragora directory."
        ),
    )
    parser.add_argument("--examples", type=int, default=10, help="Examples per Markdown category")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = repo_root(Path(args.repo))
    patch_budget = (
        None
        if args.patch_equivalence_time_budget_seconds < 0
        else args.patch_equivalence_time_budget_seconds
    )
    state_root = args.state_root.expanduser() if args.state_root else None
    outbox_dir = args.outbox_dir
    receipt_dir = args.receipt_dir
    if state_root is not None:
        if outbox_dir is None:
            outbox_dir = _automation_state_default_path(state_root, DEFAULT_OUTBOX_DIR)
        if receipt_dir is None:
            receipt_dir = _automation_state_default_path(state_root, DEFAULT_RECEIPT_DIR)
    payload = audit(
        root=root,
        base=args.base,
        repo=args.github_repo,
        prefix=args.prefix,
        recent_hours=args.recent_hours,
        max_branches=args.max_branches,
        include_patch_equivalence=args.include_patch_equivalence,
        patch_equivalence_time_budget_seconds=patch_budget,
        publisher_backlog_limit=args.publisher_backlog_limit,
        outbox_dir=outbox_dir,
        receipt_dir=receipt_dir,
    )
    if args.summary_only:
        payload = summary_only_payload(payload)
    if args.markdown:
        print_markdown(payload, examples=args.examples)
    else:
        # JSON is the default to make automation consumption explicit.
        print(json.dumps(payload, indent=2 if args.json else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
