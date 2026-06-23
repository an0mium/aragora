#!/usr/bin/env python3
"""Read-only value inventory for Aragora worktrees.

This script classifies local checkouts under the canonical Aragora worktree
directory ``<repo>/.worktrees/codex-auto`` and the legacy Codex Desktop
location ``~/.codex/worktrees`` so automation can harvest useful work before
any cleanup attempt. It never removes paths or branches.

By default both roots are scanned when present.  Pass ``--root <path>`` to
inventory a single explicit root instead.  ``--root`` may be repeated to
union multiple custom roots in one run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_codex_branch_backlog import (  # noqa: E402
    DEFAULT_OUTBOX_DIR,
    DEFAULT_RECEIPT_DIR,
    TERMINAL_RECEIPT_STATUSES,
    _commit_prefix_matches,
    is_patch_equivalent,
    terminal_receipted_handoff_branch_heads,
    unresolved_outbox_handoff_branches,
)

SCHEMA = "aragora-worktree-harvest/1.0"
# Every git/gh subprocess in this module must carry an explicit timeout so a
# wedged candidate repo (e.g. `git status` blocked on an fsmonitor daemon or
# hook) can never hang the whole inventory. Overridable per-run via
# --git-timeout-seconds (alias of the long-standing --git-timeout flag).
GIT_TIMEOUT_SECONDS = 30
MAX_SMART_MERGE_PATCH_COMMITS = 25
# Substring run_cmd embeds in stderr on timeout; classify_candidate uses it to
# annotate timed-out candidates as inspect_timeout (always protected).
_TIMEOUT_ERROR_MARKER = "timed out after"
DEFAULT_LEGACY_ROOT = Path.home() / ".codex" / "worktrees"
DEFAULT_CANONICAL_REL_ROOT = Path(".worktrees") / "codex-auto"
DEFAULT_ROOT = DEFAULT_LEGACY_ROOT  # kept for backward compatibility
DEFAULT_LEDGER_ROOT = Path(".aragora/worktree-harvest")
DEFAULT_HARVEST_RECEIPT_REL_DIR = DEFAULT_LEDGER_ROOT / "harvest-receipts"
ACTIVE_SESSION_FILES = (
    ".claude-session-active",
    ".codex_session_active",
    ".nomic-session-active",
)
RECEIPT_PATH_KEYS = frozenset(
    {
        "candidate_path",
        "candidate_repo_path",
        "checkout_path",
        "path",
        "repo_path",
        "source_path",
        "source_repo_path",
        "worktree",
        "worktree_path",
    }
)
RECEIPT_HEAD_KEYS = frozenset(
    {
        "candidate_head",
        "candidate_head_sha",
        "commit",
        "head",
        "head_sha",
        "sha",
        "source_head",
        "source_head_sha",
    }
)
TERMINAL_HARVEST_DECISION_PREFIXES = ("already_", "preserve_")
PROJECT_MARKER_FILES = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "deno.json",
    "requirements.txt",
)
VALUE_CLASSES = (
    "active_or_dirty",
    "open_pr_or_outbox",
    "receipt_protected",
    "unique_unharvested",
    "patch_equivalent_or_merged",
    "unregistered_git_residue",
    "no_git_cache_residue",
    "lookup_failed",
)
CLEANUP_CLASSES = {
    "patch_equivalent_or_merged",
    "unregistered_git_residue",
    "no_git_cache_residue",
}
PROTECTED_CLASSES = {
    "active_or_dirty",
    "open_pr_or_outbox",
    "receipt_protected",
    "lookup_failed",
}
SAFETY_CLASSES = (
    "owned",
    "unsafe_to_delete",
    "unknown_preserve",
    "referenced_preserve",
    "harvested_or_duplicate",
    "stale_or_merged",
    "stale_residue",
)


@dataclass(frozen=True)
class WorktreeEntry:
    path: Path
    branch: str | None


@dataclass
class GitInfo:
    is_repo: bool = False
    repo_path: str | None = None
    registered_worktree: bool = False
    branch: str | None = None
    head: str | None = None
    ahead: int | None = None
    behind: int | None = None
    dirty: bool = False
    patch_equivalent_to_base: bool = False
    smart_merge_equivalent_to_base: bool = False
    lookup_failed: bool = False
    inspect_timeout: bool = False
    lookup_errors: list[str] = field(default_factory=list)


@dataclass
class CleanupSafety:
    safety_class: str
    preserve: bool
    safe_to_delete: bool
    requires_live_cleanup_inspect: bool
    reason: str
    next_action: str
    signals: list[str] = field(default_factory=list)


@dataclass
class WorktreeCandidate:
    candidate_id: str
    path: str
    repo_path: str | None
    size_bytes: int | None
    size_lookup_failed: bool
    mtime: str | None
    classification: str
    decision: str
    cleanup_candidate: bool
    cleanup_safety: CleanupSafety
    proof: list[str]
    active_session: bool
    lock_files: list[str]
    git: GitInfo
    links: dict[str, Any]
    next_action: str


@dataclass
class InventoryContext:
    repo: Path
    base: str
    base_sha: str | None
    repo_remote_urls: set[str]
    strict_repo_identity: bool
    outbox_dir: Path
    receipt_dir: Path
    worktrees_by_path: dict[str, WorktreeEntry]
    unresolved_outbox_branches: set[str]
    terminal_receipt_branch_heads: dict[str, set[str | None]]
    skip_gh: bool
    git_timeout: int
    gh_timeout: int
    patch_timeout: int
    smart_merge_detection: bool = False
    smart_merge_main_subjects: list[str] = field(default_factory=list)
    open_pr_heads_cache: dict[str, list[dict[str, Any]]] | None = None
    open_pr_records_cache: list[dict[str, Any]] | None = None
    branch_pr_records_cache: dict[str, list[dict[str, Any]]] | None = None
    terminal_receipt_path_heads: dict[str, set[str | None]] = field(default_factory=dict)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill the child and its whole session, then drain pipes boundedly.

    ``subprocess.run(timeout=...)`` kills only the direct child and then
    drains its pipes with no timeout; a descendant process (git hook,
    fsmonitor daemon) that inherited the pipe FDs keeps them open and the
    drain blocks forever -- the observed "hung inside candidate git status"
    failure. Killing the whole session and bounding the drain guarantees
    run_cmd always returns.

    ``os.killpg``/``start_new_session`` are POSIX-only; on platforms without
    them the AttributeError/OSError fallback kills the direct child only.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, AttributeError):
        proc.kill()
    try:
        proc.communicate(timeout=5)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        # Last resort: abandon the pipes rather than block the inventory.
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        try:
            proc.wait(timeout=5)  # reap; SIGKILL was already sent above
        except (subprocess.TimeoutExpired, OSError):
            pass


def run_cmd(
    args: list[str],
    cwd: Path,
    *,
    timeout: int,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(args=args, returncode=124, stdout="", stderr=str(exc))
    try:
        if input_text is None:
            stdout, stderr = proc.communicate(timeout=timeout)
        else:
            stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout="",
            stderr=f"command timed out after {timeout}s: {' '.join(args)}",
        )
    except (UnicodeError, OSError, ValueError) as exc:
        _kill_process_tree(proc)
        return subprocess.CompletedProcess(
            args=args,
            returncode=125,
            stdout="",
            stderr=f"command failed while reading output: {type(exc).__name__}: {exc}",
        )
    return subprocess.CompletedProcess(
        args=args, returncode=proc.returncode, stdout=stdout or "", stderr=stderr or ""
    )


def run_git(
    args: list[str], cwd: Path, *, timeout: int = GIT_TIMEOUT_SECONDS
) -> subprocess.CompletedProcess[str]:
    return run_cmd(["git", *args], cwd, timeout=timeout)


def resolve_repo(path: Path) -> Path:
    proc = run_git(["rev-parse", "--show-toplevel"], path)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or f"not a git repository: {path}")
    return Path(proc.stdout.strip()).resolve()


def resolve_ref(repo: Path, ref: str, *, timeout: int = 30) -> str | None:
    proc = run_git(["rev-parse", "--verify", ref], repo, timeout=timeout)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def normalize_remote_url(url: str) -> str:
    value = url.strip()
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@"):
        host_path = value.removeprefix("git@").replace(":", "/", 1)
        value = f"https://{host_path}"
    if value.startswith("ssh://git@"):
        value = f"https://{value.removeprefix('ssh://git@')}"
    return value.rstrip("/").lower()


def repo_remote_urls(repo: Path, *, timeout: int = 30) -> set[str]:
    proc = run_git(["config", "--get-regexp", r"^remote\..*\.url$"], repo, timeout=timeout)
    if proc.returncode != 0:
        return set()
    urls: set[str] = set()
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        urls.add(normalize_remote_url(parts[1]))
    return urls


def parse_worktree_list(repo: Path, *, timeout: int = 30) -> dict[str, WorktreeEntry]:
    proc = run_git(["worktree", "list", "--porcelain"], repo, timeout=timeout)
    if proc.returncode != 0:
        return {}

    entries: dict[str, WorktreeEntry] = {}
    current_path: Path | None = None
    current_branch: str | None = None

    def flush() -> None:
        if current_path is None:
            return
        entries[str(current_path.resolve())] = WorktreeEntry(
            path=current_path.resolve(),
            branch=current_branch,
        )

    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            flush()
            current_path = Path(line.removeprefix("worktree ").strip())
            current_branch = None
        elif line.startswith("branch "):
            current_branch = line.removeprefix("branch refs/heads/").strip()
    flush()
    return entries


def json_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.glob("*.json") if item.is_file())


def load_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def branch_matches_receipt(
    branch: str | None,
    head: str | None,
    receipt_branch_heads: dict[str, set[str | None]],
) -> bool:
    if not branch:
        return False
    heads = receipt_branch_heads.get(branch, set())
    if not heads:
        return False
    if not head:
        return None in heads
    return any(
        receipt_head is None or _commit_prefix_matches(receipt_head, head) for receipt_head in heads
    )


def _absolute_path_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        return None
    return str(path.resolve(strict=False))


def _receipt_heads_from_mapping(payload: dict[str, Any]) -> set[str | None]:
    heads: set[str | None] = set()
    for key, value in payload.items():
        if key not in RECEIPT_HEAD_KEYS:
            continue
        if value is None:
            heads.add(None)
            continue
        if isinstance(value, str):
            text = value.strip()
            if text:
                heads.add(text)
    return heads


def _receipt_path_head_pairs(
    value: Any,
    *,
    inherited_heads: set[str | None] | None = None,
) -> list[tuple[str, str | None]]:
    heads = set(inherited_heads or set())
    pairs: list[tuple[str, str | None]] = []

    if isinstance(value, dict):
        local_heads = _receipt_heads_from_mapping(value)
        if local_heads:
            heads = local_heads
        for key, item in value.items():
            if key in RECEIPT_PATH_KEYS:
                path_key = _absolute_path_key(item)
                if path_key:
                    for head in heads or {None}:
                        pairs.append((path_key, head))
            if isinstance(item, (dict, list)):
                pairs.extend(_receipt_path_head_pairs(item, inherited_heads=heads))
    elif isinstance(value, list):
        for item in value:
            pairs.extend(_receipt_path_head_pairs(item, inherited_heads=heads))

    return pairs


def _terminal_path_receipt(payload: dict[str, Any]) -> bool:
    status = str(payload.get("status") or "").strip()
    if status in TERMINAL_RECEIPT_STATUSES:
        return True
    decision_value = payload.get("decision")
    if isinstance(decision_value, dict):
        decision = str(
            decision_value.get("outcome")
            or decision_value.get("status")
            or decision_value.get("decision")
            or ""
        ).strip()
    else:
        decision = str(decision_value or payload.get("outcome") or "").strip()
    if decision in TERMINAL_RECEIPT_STATUSES:
        return True
    if decision.startswith(TERMINAL_HARVEST_DECISION_PREFIXES):
        return True
    return False


def terminal_receipt_path_heads(receipt_roots: list[Path]) -> dict[str, set[str | None]]:
    """Return terminal receipt path refs with optional exact-head evidence."""

    refs: dict[str, set[str | None]] = {}
    for receipt_root in receipt_roots:
        for receipt_file in json_files(receipt_root):
            payload = load_json_mapping(receipt_file)
            if payload is None or not _terminal_path_receipt(payload):
                continue
            for path_key, head in _receipt_path_head_pairs(payload):
                refs.setdefault(path_key, set()).add(head)
    return refs


def path_matches_receipt(
    candidate_path: Path,
    repo_path: Path | None,
    head: str | None,
    receipt_path_heads: dict[str, set[str | None]],
) -> bool:
    path_keys = {_absolute_path_key(str(candidate_path))}
    if repo_path is not None:
        path_keys.add(_absolute_path_key(str(repo_path)))
    for path_key in {item for item in path_keys if item}:
        heads = receipt_path_heads.get(path_key, set())
        if not heads:
            continue
        if not head:
            if None in heads:
                return True
            continue
        if any(
            receipt_head is None or _commit_prefix_matches(receipt_head, head)
            for receipt_head in heads
        ):
            return True
    return False


def outbox_files_for_branch(outbox_dir: Path, branch: str | None) -> list[str]:
    if not branch:
        return []
    matches: list[str] = []
    for path in json_files(outbox_dir):
        payload = load_json_mapping(path)
        if payload is None:
            continue
        text = json.dumps(payload, sort_keys=True)
        if branch in text:
            matches.append(str(path))
    return matches


def receipt_files_for_branch(receipt_dir: Path, branch: str | None) -> list[str]:
    if not branch:
        return []
    matches: list[str] = []
    for path in json_files(receipt_dir):
        payload = load_json_mapping(path)
        if payload is None:
            continue
        text = json.dumps(payload, sort_keys=True)
        if branch in text:
            matches.append(str(path))
    return matches


def find_repo_path(candidate_root: Path) -> Path | None:
    for path in (candidate_root, candidate_root / "aragora"):
        if (path / ".git").exists():
            return path
    try:
        children = sorted(item for item in candidate_root.iterdir() if item.is_dir())
    except OSError:
        return None
    for path in children:
        if (path / ".git").exists():
            return path
    return None


def repo_identity_matches_target(
    repo_path: Path,
    *,
    context: InventoryContext,
    registered: WorktreeEntry | None,
) -> bool:
    if repo_path.resolve() == context.repo.resolve():
        return True
    if registered is not None:
        return True
    candidate_urls = repo_remote_urls(repo_path, timeout=context.git_timeout)
    return bool(
        candidate_urls and context.repo_remote_urls and candidate_urls & context.repo_remote_urls
    )


def project_marker_paths(candidate_root: Path) -> list[str]:
    try:
        roots = [candidate_root, *(item for item in candidate_root.iterdir() if item.is_dir())]
    except OSError:
        return [str(candidate_root)]
    markers: list[str] = []
    for root in roots:
        for marker in PROJECT_MARKER_FILES:
            if (root / marker).exists():
                markers.append(str(root / marker))
    return sorted(markers)


def active_lock_files(candidate_root: Path, repo_path: Path | None) -> list[str]:
    roots = [candidate_root]
    if repo_path is not None and repo_path != candidate_root:
        roots.append(repo_path)
    found: list[str] = []
    for root in roots:
        for name in ACTIVE_SESSION_FILES:
            if (root / name).exists():
                found.append(str(root / name))
    return sorted(found)


def has_active_session(candidate_root: Path, repo_path: Path | None) -> bool:
    return bool(active_lock_files(candidate_root, repo_path))


def git_status_dirty(repo_path: Path, *, timeout: int) -> tuple[bool, bool, str | None]:
    proc = run_git(["status", "--porcelain"], repo_path, timeout=timeout)
    if proc.returncode != 0:
        return True, True, proc.stderr.strip() or "git status failed"
    return bool(proc.stdout.strip()), False, None


def git_branch(
    repo_path: Path, registered: WorktreeEntry | None, *, timeout: int
) -> tuple[str | None, bool, str | None]:
    if registered and registered.branch:
        return registered.branch, False, None
    proc = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_path, timeout=timeout)
    if proc.returncode != 0:
        return None, True, proc.stderr.strip() or "branch lookup failed"
    branch = proc.stdout.strip()
    if not branch or branch == "HEAD":
        return None, False, None
    return branch, False, None


def git_head(repo_path: Path, *, timeout: int) -> tuple[str | None, bool, str | None]:
    proc = run_git(["rev-parse", "HEAD"], repo_path, timeout=timeout)
    if proc.returncode != 0:
        return None, True, proc.stderr.strip() or "head lookup failed"
    return proc.stdout.strip() or None, False, None


def git_ahead_behind(
    repo_path: Path, base: str, rev: str, *, timeout: int
) -> tuple[int | None, int | None, bool, str | None]:
    proc = run_git(
        ["rev-list", "--left-right", "--count", f"{base}...{rev}"], repo_path, timeout=timeout
    )
    if proc.returncode != 0:
        return None, None, True, proc.stderr.strip() or "ahead/behind lookup failed"
    try:
        behind_text, ahead_text = proc.stdout.split()
        return int(ahead_text), int(behind_text), False, None
    except ValueError:
        return None, None, True, f"unexpected ahead/behind output: {proc.stdout!r}"


def normalize_commit_subject(subject: str) -> str:
    """Normalize commit/PR titles for loose squash-merge equivalence checks."""

    value = subject.lower()
    value = re.sub(r"\s+\(#\d+\)\s*$", "", value)
    value = re.sub(r"\s+\[lane:[^\]]+\]\s*$", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def commit_subject_matches_recent_main(
    subject: str,
    recent_main_subjects: list[str],
    *,
    threshold: float = 0.80,
) -> bool:
    normalized = normalize_commit_subject(subject)
    if not normalized:
        return False
    for recent_subject in recent_main_subjects:
        recent = normalize_commit_subject(recent_subject)
        if not recent:
            continue
        if normalized in recent or recent in normalized:
            return True
        if SequenceMatcher(None, normalized, recent).ratio() >= threshold:
            return True
    return False


def recent_main_commit_subjects(repo: Path, base: str, *, timeout: int) -> list[str]:
    proc = run_git(["log", base, "--since=60 days", "--pretty=format:%s"], repo, timeout=timeout)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def branch_unique_commit_subjects(
    repo_path: Path,
    base: str,
    rev: str,
    *,
    timeout: int,
) -> list[str] | None:
    proc = run_git(
        ["log", "--no-merges", f"{base}..{rev}", "--pretty=format:%s"],
        repo_path,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return None
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def branch_subjects_match_recent_main(
    repo_path: Path,
    base: str,
    rev: str,
    recent_main_subjects: list[str],
    *,
    timeout: int,
) -> tuple[bool, list[str]]:
    subjects = branch_unique_commit_subjects(repo_path, base, rev, timeout=timeout)
    if not subjects:
        return False, []
    matched = [
        subject
        for subject in subjects
        if commit_subject_matches_recent_main(subject, recent_main_subjects)
    ]
    return len(matched) == len(subjects), matched


def branch_patches_present_on_base(
    repo_path: Path,
    base: str,
    rev: str,
    *,
    timeout: int,
    lookup_errors: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Return true when every unique non-merge commit patch is already in base.

    Some stale branches contain local commits whose changes were later merged
    through a different PR/commit, while the stale branch's old tree still
    differs from current main. Plain tree-diff or `git cherry` checks can leave
    those as false `unique_unharvested` rows. Use a temporary index loaded with
    `base` and verify each commit's reverse patch applies there. This is the
    same safety property as "cherry-pick onto base would be empty", without
    mutating the candidate worktree.
    """

    commits_proc = run_git(
        ["rev-list", "--reverse", "--no-merges", f"{base}..{rev}"],
        repo_path,
        timeout=timeout,
    )
    if commits_proc.returncode != 0:
        if lookup_errors is not None:
            detail = (commits_proc.stderr or commits_proc.stdout or "").strip()
            lookup_errors.append(
                f"patch-present rev-list failed: {detail or commits_proc.returncode}"
            )
        return False, []
    commits = [line.strip() for line in commits_proc.stdout.splitlines() if line.strip()]
    if not commits:
        return False, []
    if len(commits) > MAX_SMART_MERGE_PATCH_COMMITS:
        if lookup_errors is not None:
            lookup_errors.append(
                "patch-present skipped: "
                f"{len(commits)} commits exceeds budget {MAX_SMART_MERGE_PATCH_COMMITS}"
            )
        return False, []

    fd, index_path = tempfile.mkstemp(prefix="aragora-inventory-index-")
    os.close(fd)
    try:
        env = dict(os.environ)
        env["GIT_INDEX_FILE"] = index_path
        read_tree = run_cmd(["git", "read-tree", base], repo_path, timeout=timeout, env=env)
        if read_tree.returncode != 0:
            if lookup_errors is not None:
                detail = (read_tree.stderr or read_tree.stdout or "").strip()
                lookup_errors.append(
                    f"patch-present read-tree failed: {detail or read_tree.returncode}"
                )
            return False, []

        verified: list[str] = []
        for commit in commits:
            patch = run_git(["show", "--format=", "--binary", commit], repo_path, timeout=timeout)
            if patch.returncode != 0 or not patch.stdout.strip():
                if lookup_errors is not None:
                    detail = (patch.stderr or patch.stdout or "").strip()
                    reason = "empty patch" if patch.returncode == 0 else detail or patch.returncode
                    lookup_errors.append(f"patch-present show failed for {commit}: {reason}")
                return False, verified
            reverse_check = run_cmd(
                ["git", "apply", "--cached", "--reverse", "--check", "-"],
                repo_path,
                timeout=timeout,
                env=env,
                input_text=patch.stdout,
            )
            if reverse_check.returncode != 0:
                if lookup_errors is not None and reverse_check.returncode == 124:
                    detail = (reverse_check.stderr or reverse_check.stdout or "").strip()
                    lookup_errors.append(
                        f"patch-present reverse-check failed for {commit}: "
                        f"{detail or reverse_check.returncode}"
                    )
                return False, verified
            verified.append(commit)
        return True, verified
    finally:
        try:
            os.unlink(index_path)
        except FileNotFoundError:
            pass


def branch_unique_merge_commits(
    repo_path: Path,
    base: str,
    rev: str,
    *,
    timeout: int,
) -> tuple[list[str] | None, str | None]:
    proc = run_git(["rev-list", "--merges", f"{base}..{rev}"], repo_path, timeout=timeout)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        return None, detail or f"git rev-list --merges exited {proc.returncode}"
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()], None


def branch_merge_tree_matches_base(
    repo_path: Path,
    base: str,
    rev: str,
    *,
    timeout: int,
) -> tuple[bool | None, str | None]:
    """Return true when merging rev into base would leave base's tree unchanged."""

    base_tree = run_git(["rev-parse", f"{base}^{{tree}}"], repo_path, timeout=timeout)
    if base_tree.returncode != 0:
        detail = (base_tree.stderr or base_tree.stdout or "").strip()
        return None, detail or f"git rev-parse {base}^{{tree}} exited {base_tree.returncode}"
    base_tree_sha = base_tree.stdout.strip().splitlines()[0] if base_tree.stdout.strip() else ""
    if not base_tree_sha:
        return None, "git rev-parse returned an empty tree id"

    merge_tree = run_git(
        ["merge-tree", "--write-tree", "--no-messages", base, rev],
        repo_path,
        timeout=timeout,
    )
    if merge_tree.returncode != 0:
        detail = (merge_tree.stderr or merge_tree.stdout or "").strip()
        reason = detail or f"git merge-tree exited {merge_tree.returncode}"
        if merge_tree.returncode == 124 or _TIMEOUT_ERROR_MARKER in reason:
            return None, reason
        return False, reason
    merged_tree_sha = merge_tree.stdout.strip().splitlines()[0] if merge_tree.stdout.strip() else ""
    if not merged_tree_sha:
        return None, "git merge-tree returned an empty tree id"
    return merged_tree_sha == base_tree_sha, None


def prefetch_open_pr_heads(
    repo: Path, *, timeout: int
) -> tuple[
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    bool,
    str | None,
]:
    open_proc = run_cmd(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number,title,url,headRefName,body,state,headRefOid",
        ],
        repo,
        timeout=timeout,
    )
    if open_proc.returncode != 0:
        return {}, [], {}, True, open_proc.stderr.strip() or "gh pr open prefetch failed"
    try:
        open_payload = json.loads(open_proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        return {}, [], {}, True, f"failed to parse gh pr open prefetch output: {exc}"
    if not isinstance(open_payload, list):
        return {}, [], {}, True, "gh pr open prefetch output was not a list"

    all_proc = run_cmd(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--limit",
            "500",
            "--json",
            "number,title,url,headRefName,body,state,headRefOid",
        ],
        repo,
        timeout=timeout,
    )
    if all_proc.returncode != 0:
        return {}, [], {}, True, all_proc.stderr.strip() or "gh pr all-state prefetch failed"
    try:
        all_payload = json.loads(all_proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        return {}, [], {}, True, f"failed to parse gh pr all-state prefetch output: {exc}"
    if not isinstance(all_payload, list):
        return {}, [], {}, True, "gh pr all-state prefetch output was not a list"

    cache: dict[str, list[dict[str, Any]]] = {}
    records: list[dict[str, Any]] = []
    branch_records: dict[str, list[dict[str, Any]]] = {}
    for item in all_payload:
        if not isinstance(item, dict):
            continue
        head = item.get("headRefName")
        if not isinstance(head, str) or not head:
            continue
        record = {
            k: v
            for k, v in item.items()
            if k in ("number", "title", "url", "headRefName", "body", "state", "headRefOid")
        }
        branch_records.setdefault(head, []).append(record)
    for item in open_payload:
        if not isinstance(item, dict):
            continue
        head = item.get("headRefName")
        if not isinstance(head, str) or not head:
            continue
        if str(item.get("state") or "OPEN").upper() != "OPEN":
            continue
        record = {
            k: v
            for k, v in item.items()
            if k in ("number", "title", "url", "headRefName", "body", "state", "headRefOid")
        }
        records.append(record)
        cache.setdefault(head, []).append(record)
    return cache, records, branch_records, False, None


def lookup_open_prs(
    repo: Path,
    branch: str | None,
    *,
    timeout: int,
    skip_gh: bool,
    cached_open_pr_heads: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    if not branch:
        return [], False, None
    if cached_open_pr_heads is not None:
        return list(cached_open_pr_heads.get(branch, [])), False, None
    if skip_gh:
        return [], False, None
    proc = run_cmd(
        ["gh", "pr", "list", "--state", "open", "--head", branch, "--json", "number,title,url"],
        repo,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return [], True, proc.stderr.strip() or "gh pr lookup failed"
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        return [], True, f"failed to parse gh pr output: {exc}"
    if not isinstance(payload, list):
        return [], True, "gh pr output was not a list"
    return [item for item in payload if isinstance(item, dict)], False, None


def lookup_branch_prs(
    repo: Path,
    branch: str | None,
    *,
    timeout: int,
    skip_gh: bool,
    cached_branch_prs: dict[str, list[dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    """Return all GitHub PR records for a branch.

    This is used only for preserve decisions: if a stale closed PR branch is
    explicitly superseded by an open PR, inventory must not propose harvesting
    that old branch again.
    """

    if not branch or skip_gh:
        return [], False, None
    if cached_branch_prs is not None:
        return list(cached_branch_prs.get(branch, [])), False, None
    proc = run_cmd(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--head",
            branch,
            "--json",
            "number,title,url,state,headRefName,headRefOid",
        ],
        repo,
        timeout=timeout,
    )
    if proc.returncode != 0:
        return [], True, proc.stderr.strip() or "gh pr branch lookup failed"
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        return [], True, f"failed to parse gh pr branch output: {exc}"
    if not isinstance(payload, list):
        return [], True, "gh pr branch output was not a list"
    return [item for item in payload if isinstance(item, dict)], False, None


_SUPERSESSION_TERMS = (
    "supersede",
    "supersedes",
    "superseded",
    "re-cut",
    "recut",
    "re cut",
    "replaces",
    "replacement",
)


def _open_pr_explicitly_supersedes(source_pr_number: int, open_pr: dict[str, Any]) -> bool:
    text = f"{open_pr.get('title') or ''}\n{open_pr.get('body') or ''}".lower()
    if not text:
        return False
    for match in re.finditer(rf"#\s*{source_pr_number}\b", text):
        window = text[max(0, match.start() - 96) : min(len(text), match.end() + 96)]
        if any(term in window for term in _SUPERSESSION_TERMS):
            return True
    return False


def lookup_superseding_open_prs(
    repo: Path,
    branch: str | None,
    *,
    timeout: int,
    skip_gh: bool,
    open_pr_records: list[dict[str, Any]] | None,
    branch_pr_records: dict[str, list[dict[str, Any]]] | None,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    if not branch or skip_gh or not open_pr_records:
        return [], False, None

    branch_prs, failed, error = lookup_branch_prs(
        repo,
        branch,
        timeout=timeout,
        skip_gh=skip_gh,
        cached_branch_prs=branch_pr_records,
    )
    if failed:
        return [], True, error

    source_numbers: list[int] = []
    for item in branch_prs:
        state = str(item.get("state") or "").upper()
        number = item.get("number")
        if state == "OPEN" or not isinstance(number, int):
            continue
        source_numbers.append(number)

    superseding: list[dict[str, Any]] = []
    for source_number in source_numbers:
        for open_pr in open_pr_records:
            if _open_pr_explicitly_supersedes(source_number, open_pr):
                record = {
                    k: v
                    for k, v in open_pr.items()
                    if k in ("number", "title", "url", "headRefName")
                }
                record["supersedes_pr"] = source_number
                superseding.append(record)
    return superseding, False, None


def measure_sizes(
    paths: list[Path], *, mode: str, timeout: int
) -> tuple[dict[str, int | None], set[str]]:
    if mode == "none":
        return {str(path): None for path in paths}, set()
    if not paths:
        return {}, set()
    if mode == "stat":
        sizes: dict[str, int | None] = {}
        stat_failed: set[str] = set()
        for path in paths:
            try:
                sizes[str(path)] = path.stat().st_blocks * 512
            except OSError:
                sizes[str(path)] = None
                stat_failed.add(str(path))
        return sizes, stat_failed

    proc = run_cmd(["du", "-sk", *[str(path) for path in paths]], Path("/"), timeout=timeout)
    sizes = {str(path): None for path in paths}
    du_failed: set[str] = set()
    if proc.returncode != 0:
        return sizes, {str(path) for path in paths}
    for line in proc.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            sizes[str(Path(parts[1]))] = int(parts[0]) * 1024
        except ValueError:
            du_failed.add(str(Path(parts[1])))
    for path_text, size in sizes.items():
        if size is None:
            du_failed.add(path_text)
    return sizes, du_failed


def candidate_id(path: Path, repo_path: Path | None) -> str:
    raw = f"{path.resolve()}|{repo_path.resolve() if repo_path else ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def candidate_mtime(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0).isoformat()
    except OSError:
        return None


def classify_candidate(
    candidate_root: Path,
    *,
    context: InventoryContext,
    size_bytes: int | None,
    size_lookup_failed: bool,
) -> WorktreeCandidate:
    repo_path = find_repo_path(candidate_root)
    active_session = has_active_session(candidate_root, repo_path)
    lock_files = active_lock_files(candidate_root, repo_path)
    git = GitInfo(is_repo=repo_path is not None, repo_path=str(repo_path) if repo_path else None)
    proof: list[str] = []
    links: dict[str, Any] = {
        "open_prs": [],
        "superseding_open_prs": [],
        "outbox_files": [],
        "receipt_files": [],
    }

    if repo_path is None:
        if active_session or lock_files:
            classification = "active_or_dirty"
            proof.append("active session marker present without git metadata")
        elif context.strict_repo_identity and (
            project_markers := project_marker_paths(candidate_root)
        ):
            classification = "lookup_failed"
            git.lookup_failed = True
            git.lookup_errors.append("project markers exist without confirmed Aragora git metadata")
            proof.append("project-like directory is not confirmed as Aragora")
            links["project_markers"] = project_markers
        else:
            classification = "no_git_cache_residue"
            proof.append("no git metadata at candidate root or candidate/aragora path")
        return build_candidate(
            candidate_root,
            repo_path,
            size_bytes,
            size_lookup_failed,
            classification,
            active_session,
            lock_files,
            git,
            links,
            proof,
        )

    registered = context.worktrees_by_path.get(str(repo_path.resolve()))
    git.registered_worktree = registered is not None

    if context.strict_repo_identity and not repo_identity_matches_target(
        repo_path,
        context=context,
        registered=registered,
    ):
        git.lookup_failed = True
        git.lookup_errors.append("repo identity does not match target repo")
        return build_candidate(
            candidate_root,
            repo_path,
            size_bytes,
            size_lookup_failed,
            "lookup_failed",
            active_session,
            lock_files,
            git,
            links,
            ["repo identity does not match target repo"],
        )

    branch, branch_failed, branch_error = git_branch(
        repo_path, registered, timeout=context.git_timeout
    )
    git.branch = branch
    if branch_failed:
        git.lookup_failed = True
        git.lookup_errors.append(branch_error or "branch lookup failed")
    head, head_failed, head_error = git_head(repo_path, timeout=context.git_timeout)
    git.head = head
    if head_failed:
        git.lookup_failed = True
        git.lookup_errors.append(head_error or "head lookup failed")

    dirty, dirty_failed, dirty_error = git_status_dirty(repo_path, timeout=context.git_timeout)
    git.dirty = dirty
    if dirty_failed:
        git.lookup_failed = True
        git.lookup_errors.append(dirty_error or "git status failed")

    rev = branch or head
    if rev and context.base_sha is not None:
        ahead, behind, divergence_failed, divergence_error = git_ahead_behind(
            repo_path,
            context.base,
            rev,
            timeout=context.git_timeout,
        )
        git.ahead = ahead
        git.behind = behind
        if divergence_failed:
            git.lookup_failed = True
            git.lookup_errors.append(divergence_error or "ahead/behind lookup failed")
    elif rev:
        git.lookup_failed = True
        git.lookup_errors.append(f"base ref not found: {context.base}")

    open_prs, open_pr_failed, open_pr_error = lookup_open_prs(
        context.repo,
        branch,
        timeout=context.gh_timeout,
        skip_gh=context.skip_gh,
        cached_open_pr_heads=context.open_pr_heads_cache,
    )
    links["open_prs"] = open_prs
    if open_pr_failed:
        git.lookup_failed = True
        git.lookup_errors.append(open_pr_error or "open PR lookup failed")

    links["outbox_files"] = outbox_files_for_branch(context.outbox_dir, branch)
    links["receipt_files"] = receipt_files_for_branch(context.receipt_dir, branch)
    outbox_protected = bool(branch and branch in context.unresolved_outbox_branches)
    branch_receipt_protected = branch_matches_receipt(
        branch,
        head,
        context.terminal_receipt_branch_heads,
    )
    path_receipt_protected = path_matches_receipt(
        candidate_root,
        repo_path,
        head,
        context.terminal_receipt_path_heads,
    )
    receipt_protected = branch_receipt_protected or path_receipt_protected

    if active_session or lock_files or dirty:
        classification = "active_or_dirty"
        if active_session:
            proof.append("active session marker present")
        if lock_files:
            proof.append("active lock file present")
        if dirty:
            proof.append("git status is dirty or unavailable")
    elif git.lookup_failed:
        classification = "lookup_failed"
        proof.extend(git.lookup_errors or ["one or more git/GitHub lookups failed"])
    elif open_prs or outbox_protected:
        classification = "open_pr_or_outbox"
        if open_prs:
            proof.append("open PR exists for branch")
        if outbox_protected:
            proof.append("unresolved automation outbox references branch")
    elif receipt_protected:
        classification = "receipt_protected"
        if branch_receipt_protected:
            proof.append("terminal automation receipt references branch/head")
        if path_receipt_protected:
            proof.append("terminal receipt references path/head")
    elif git.ahead and git.ahead > 0:
        superseding_open_prs, superseding_failed, superseding_error = lookup_superseding_open_prs(
            context.repo,
            branch,
            timeout=context.gh_timeout,
            skip_gh=context.skip_gh,
            open_pr_records=context.open_pr_records_cache,
            branch_pr_records=context.branch_pr_records_cache,
        )
        links["superseding_open_prs"] = superseding_open_prs
        if superseding_failed:
            git.lookup_failed = True
            git.lookup_errors.append(superseding_error or "superseding open PR lookup failed")
            classification = "lookup_failed"
            proof.append("superseding open PR lookup failed")
        elif superseding_open_prs:
            classification = "open_pr_or_outbox"
            proof.append("open PR explicitly supersedes closed source PR for branch")
        else:
            patch_equivalent = False
            try:
                patch_equivalent = is_patch_equivalent(
                    repo_path,
                    context.base,
                    rev or "HEAD",
                    timeout=context.patch_timeout,
                )
            except Exception as exc:
                git.lookup_failed = True
                git.lookup_errors.append(f"patch equivalence failed: {exc}")
                classification = "lookup_failed"
                proof.append("patch equivalence lookup failed")
            else:
                git.patch_equivalent_to_base = patch_equivalent
                if patch_equivalent:
                    classification = "patch_equivalent_or_merged"
                    proof.append("branch is patch-equivalent to base")
                elif context.smart_merge_detection:
                    merge_tree_matches, merge_tree_error = branch_merge_tree_matches_base(
                        repo_path,
                        context.base,
                        rev or "HEAD",
                        timeout=context.patch_timeout,
                    )
                    if merge_tree_matches is None and merge_tree_error:
                        git.lookup_failed = True
                        git.lookup_errors.append(
                            f"smart merge merge-tree lookup failed: {merge_tree_error}"
                        )
                        classification = "lookup_failed"
                        proof.append("smart merge lookup failed")
                    elif merge_tree_matches:
                        git.smart_merge_equivalent_to_base = True
                        classification = "patch_equivalent_or_merged"
                        proof.append("merging branch into base leaves base tree unchanged")
                        links["smart_merge_merge_tree"] = context.base
                    elif merge_tree_error:
                        classification = "unique_unharvested"
                        proof.append(
                            "merge-tree did not prove branch is already represented on base"
                        )
                        links["smart_merge_merge_tree_error"] = merge_tree_error
                    else:
                        merge_commits, merge_error = branch_unique_merge_commits(
                            repo_path,
                            context.base,
                            rev or "HEAD",
                            timeout=context.patch_timeout,
                        )
                        if merge_error:
                            git.lookup_failed = True
                            git.lookup_errors.append(
                                f"smart merge merge-commit lookup failed: {merge_error}"
                            )
                            classification = "lookup_failed"
                            proof.append("smart merge lookup failed")
                        elif merge_commits:
                            classification = "unique_unharvested"
                            proof.append(
                                "smart merge detection skipped because branch contains merge commits"
                            )
                            links["smart_merge_merge_commits"] = merge_commits
                        else:
                            smart_equivalent, matched_subjects = branch_subjects_match_recent_main(
                                repo_path,
                                context.base,
                                rev or "HEAD",
                                context.smart_merge_main_subjects,
                                timeout=context.patch_timeout,
                            )
                            if smart_equivalent:
                                proof.append(
                                    "all unique commit subjects match recent main squash-merge "
                                    "subjects (advisory; patch proof still required)"
                                )
                                links["smart_merge_matched_subjects"] = matched_subjects
                            error_count_before = len(git.lookup_errors)
                            patches_present, matched_commits = branch_patches_present_on_base(
                                repo_path,
                                context.base,
                                rev or "HEAD",
                                timeout=context.patch_timeout,
                                lookup_errors=git.lookup_errors,
                            )
                            git.smart_merge_equivalent_to_base = patches_present
                            if patches_present:
                                classification = "patch_equivalent_or_merged"
                                proof.append(
                                    "all unique commit patches are already present on base"
                                )
                                links["smart_merge_matched_commits"] = matched_commits
                            elif len(git.lookup_errors) > error_count_before:
                                git.lookup_failed = True
                                classification = "lookup_failed"
                                proof.append("smart merge patch-present lookup failed")
                            else:
                                classification = "unique_unharvested"
                                proof.append("branch has unique commits or diff ahead of base")
                else:
                    classification = "unique_unharvested"
                    proof.append("branch has unique commits or diff ahead of base")
    elif git.registered_worktree:
        classification = "patch_equivalent_or_merged"
        proof.append("registered git worktree has no unique commits ahead of base")
    else:
        classification = "unregistered_git_residue"
        proof.append("git checkout is not registered in git worktree list")

    if any(_TIMEOUT_ERROR_MARKER in error for error in git.lookup_errors):
        # A timed-out lookup is never authoritative: the candidate is already
        # routed to a protected class (active_or_dirty via the fail-dirty
        # status path, or lookup_failed), so it can never be safe-to-clean.
        # Annotate it so operators and the summary can count timeouts.
        git.inspect_timeout = True
        proof.append("inspect_timeout: a git/GitHub lookup timed out; candidate is protected")

    return build_candidate(
        candidate_root,
        repo_path,
        size_bytes,
        size_lookup_failed,
        classification,
        active_session,
        lock_files,
        git,
        links,
        proof,
    )


def build_candidate(
    candidate_root: Path,
    repo_path: Path | None,
    size_bytes: int | None,
    size_lookup_failed: bool,
    classification: str,
    active_session: bool,
    lock_files: list[str],
    git: GitInfo,
    links: dict[str, Any],
    proof: list[str],
) -> WorktreeCandidate:
    cleanup_candidate = classification in CLEANUP_CLASSES and not active_session and not git.dirty
    if classification == "unique_unharvested":
        decision = "harvest_candidate"
        next_action = "inspect diff and harvest into a fresh branch or handoff"
    elif cleanup_candidate:
        decision = "cleanup_candidate"
        next_action = "fresh safe_worktree_cleanup.py inspect is required before any removal"
    elif classification in PROTECTED_CLASSES:
        decision = "preserve"
        next_action = "preserve until blocker clears or value is harvested"
    else:
        decision = "preserve"
        next_action = "review classification before cleanup"
    cleanup_safety = cleanup_safety_for_candidate(
        classification=classification,
        decision=decision,
        cleanup_candidate=cleanup_candidate,
        active_session=active_session,
        lock_files=lock_files,
        git=git,
        links=links,
    )
    return WorktreeCandidate(
        candidate_id=candidate_id(candidate_root, repo_path),
        path=str(candidate_root),
        repo_path=str(repo_path) if repo_path else None,
        size_bytes=size_bytes,
        size_lookup_failed=size_lookup_failed,
        mtime=candidate_mtime(candidate_root),
        classification=classification,
        decision=decision,
        cleanup_candidate=cleanup_candidate,
        cleanup_safety=cleanup_safety,
        proof=proof,
        active_session=active_session,
        lock_files=lock_files,
        git=git,
        links=links,
        next_action=next_action,
    )


def cleanup_safety_for_candidate(
    *,
    classification: str,
    decision: str,
    cleanup_candidate: bool,
    active_session: bool,
    lock_files: list[str],
    git: GitInfo,
    links: dict[str, Any],
) -> CleanupSafety:
    if active_session or lock_files:
        return CleanupSafety(
            safety_class="owned",
            preserve=True,
            safe_to_delete=False,
            requires_live_cleanup_inspect=False,
            reason="active session or owner lock marker is present",
            next_action="route to the active owner or wait for explicit release",
            signals=["owned"],
        )
    if git.dirty or decision == "harvest_candidate":
        reason = (
            "branch has unique unharvested work"
            if decision == "harvest_candidate"
            else "git status is dirty or unavailable"
        )
        return CleanupSafety(
            safety_class="unsafe_to_delete",
            preserve=True,
            safe_to_delete=False,
            requires_live_cleanup_inspect=False,
            reason=reason,
            next_action="preserve and harvest or resolve dirty state before any cleanup",
            signals=["unsafe_to_delete"],
        )
    if git.lookup_failed or classification == "lookup_failed":
        return CleanupSafety(
            safety_class="unknown_preserve",
            preserve=True,
            safe_to_delete=False,
            requires_live_cleanup_inspect=False,
            reason="one or more identity, git, GitHub, or patch-equivalence lookups failed",
            next_action="preserve until lookup succeeds and a fresh cleanup inspect agrees",
            signals=["unknown", "unsafe_to_delete"],
        )
    if classification == "open_pr_or_outbox":
        signals = ["referenced"]
        if links.get("open_prs"):
            signals.append("duplicate")
        return CleanupSafety(
            safety_class="referenced_preserve",
            preserve=True,
            safe_to_delete=False,
            requires_live_cleanup_inspect=False,
            reason="open PR or unresolved outbox still references the branch",
            next_action="preserve or route to the owning publication/handoff lane",
            signals=signals,
        )
    if classification == "receipt_protected":
        return CleanupSafety(
            safety_class="referenced_preserve",
            preserve=True,
            safe_to_delete=False,
            requires_live_cleanup_inspect=False,
            reason="terminal receipt references this branch or head",
            next_action="preserve unless a bounded cleanup lane verifies supersedence",
            signals=["referenced", "harvested"],
        )
    if classification == "patch_equivalent_or_merged":
        safety_class = "stale_or_merged" if git.registered_worktree else "harvested_or_duplicate"
        return CleanupSafety(
            safety_class=safety_class,
            preserve=False,
            safe_to_delete=False,
            requires_live_cleanup_inspect=True,
            reason="branch has no unique diff against base or matches a merged patch",
            next_action="run fresh safe_worktree_cleanup.py inspect before any removal",
            signals=["stale", "harvested", "duplicate"],
        )
    if cleanup_candidate and classification in {"unregistered_git_residue", "no_git_cache_residue"}:
        return CleanupSafety(
            safety_class="stale_residue",
            preserve=False,
            safe_to_delete=False,
            requires_live_cleanup_inspect=True,
            reason="local residue has no active owner or unique confirmed work in inventory",
            next_action="run fresh safe_worktree_cleanup.py inspect before any removal",
            signals=["stale"],
        )
    return CleanupSafety(
        safety_class="unknown_preserve",
        preserve=True,
        safe_to_delete=False,
        requires_live_cleanup_inspect=False,
        reason=f"inventory classification is not cleanup-authoritative: {classification}",
        next_action="preserve until a narrower helper provides cleanup authority",
        signals=["unknown"],
    )


def candidate_roots(root: Path, limit: int | None = None) -> list[Path]:
    if not root.exists():
        return []
    entries = sorted(
        (entry for entry in root.iterdir() if entry.is_dir()),
        key=lambda path: path.name,
    )
    return entries[:limit] if limit is not None else entries


def _git_common_dir(repo: Path) -> Path | None:
    """Return the git common dir for ``repo`` without raising on non-repos."""
    # cwd="." is deliberate: ``repo`` may be a deleted/broken worktree path,
    # and Popen(cwd=<missing dir>) raises before git can answer; ``git -C``
    # reports the failure gracefully instead.
    result = run_cmd(
        ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        Path("."),
        timeout=5,
    )
    if result.returncode != 0:
        return None
    raw = result.stdout.strip()
    return Path(raw) if raw else None


def _default_canonical_root_candidates(repo: Path) -> list[Path]:
    candidates = [repo / DEFAULT_CANONICAL_REL_ROOT]
    common_dir = _git_common_dir(repo)
    if common_dir is not None and common_dir.name == ".git":
        candidates.append(common_dir.parent / DEFAULT_CANONICAL_REL_ROOT)
    return candidates


def resolve_default_roots(repo: Path) -> list[Path]:
    """Return ordered default inventory roots for ``repo``.

    Preference order:
    1. ``<repo>/.worktrees/codex-auto`` -- the canonical Aragora worktree
       directory written by ``scripts/codex_worktree_autopilot.py ensure``.
    2. ``~/.codex/worktrees`` -- the legacy Codex Desktop location, kept
       for backwards compatibility with sessions that pre-date the
       canonical move.

    Each path is included only if it exists on disk.  After ``resolve()``
    duplicates are dropped while preserving order (handles the case where
    a user's repo is symlinked under their home directory).
    """
    seen: set[str] = set()
    roots: list[Path] = []
    for candidate in _default_canonical_root_candidates(repo):
        try:
            canonical = candidate.resolve()
        except OSError:
            continue
        if canonical.exists() and str(canonical) not in seen:
            roots.append(canonical)
            seen.add(str(canonical))
    try:
        legacy = DEFAULT_LEGACY_ROOT.resolve()
    except OSError:
        legacy = None
    if legacy is not None and legacy.exists() and str(legacy) not in seen:
        roots.append(legacy)
        seen.add(str(legacy))
    return roots


def candidate_roots_from(roots: list[Path], limit: int | None = None) -> list[Path]:
    """Concatenate entries across ``roots`` in order, applying ``limit`` once.

    Used when more than one inventory root is in play (canonical + legacy
    on the same host, or multiple ``--root`` flags).  Single-root callers
    can continue to use ``candidate_roots`` for backwards compatibility.
    """
    entries: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for entry in sorted(
            (item for item in root.iterdir() if item.is_dir()),
            key=lambda path: path.name,
        ):
            key = str(entry.resolve())
            if key in seen:
                continue
            seen.add(key)
            entries.append(entry)
    return entries[:limit] if limit is not None else entries


def build_summary(candidates: list[WorktreeCandidate]) -> dict[str, Any]:
    counts = Counter(candidate.classification for candidate in candidates)
    safety_counts = Counter(candidate.cleanup_safety.safety_class for candidate in candidates)
    bytes_by_class: dict[str, int] = dict.fromkeys(VALUE_CLASSES, 0)
    known_bytes = 0
    size_lookup_failures = 0
    for candidate in candidates:
        if candidate.size_bytes is None:
            size_lookup_failures += 1
            continue
        known_bytes += candidate.size_bytes
        bytes_by_class[candidate.classification] = (
            bytes_by_class.get(candidate.classification, 0) + candidate.size_bytes
        )

    def top(filter_fn: Any) -> list[dict[str, Any]]:
        selected = [candidate for candidate in candidates if filter_fn(candidate)]
        selected.sort(key=lambda item: item.size_bytes or -1, reverse=True)
        return [
            {
                "path": candidate.path,
                "classification": candidate.classification,
                "safety_class": candidate.cleanup_safety.safety_class,
                "size_bytes": candidate.size_bytes,
                "branch": candidate.git.branch,
                "head": candidate.git.head,
                "decision": candidate.decision,
                "cleanup_safety": asdict(candidate.cleanup_safety),
                "proof": candidate.proof,
            }
            for candidate in selected[:20]
        ]

    return {
        "total_candidates": len(candidates),
        "classified_candidates": sum(counts.values()),
        "unknown_candidates": counts.get("lookup_failed", 0),
        "count_by_class": {name: counts.get(name, 0) for name in VALUE_CLASSES},
        "count_by_safety_class": {name: safety_counts.get(name, 0) for name in SAFETY_CLASSES},
        "bytes_by_class": bytes_by_class,
        "known_size_bytes": known_bytes,
        "size_lookup_failures": size_lookup_failures,
        "inspect_timeouts": sum(1 for candidate in candidates if candidate.git.inspect_timeout),
        "inventory_coverage": (
            1.0 if not candidates else (len(candidates) - size_lookup_failures) / len(candidates)
        ),
        "cleanup_candidate_count": sum(
            1 for candidate in candidates if candidate.cleanup_candidate
        ),
        "harvest_candidate_count": counts.get("unique_unharvested", 0),
        "top_protected_size_users": top(
            lambda candidate: candidate.classification in PROTECTED_CLASSES
        ),
        "top_cleanup_candidates": top(lambda candidate: candidate.cleanup_candidate),
        "top_unique_unharvested": top(
            lambda candidate: candidate.classification == "unique_unharvested"
        ),
    }


def inventory(
    *,
    root: Path | None = None,
    roots: list[Path] | None = None,
    repo: Path,
    base: str,
    outbox_dir: Path,
    receipt_dir: Path,
    limit: int | None,
    size_mode: str,
    size_timeout: int,
    skip_gh: bool,
    git_timeout: int,
    gh_timeout: int,
    patch_timeout: int,
    smart_merge_detection: bool = False,
    include_pr_state: bool = False,
) -> dict[str, Any]:
    repo = resolve_repo(repo)
    base_sha = resolve_ref(repo, base, timeout=git_timeout)
    explicit_roots = bool(root is not None or roots)
    if roots is None:
        roots = [root] if root is not None else []
    if not roots:
        roots = resolve_default_roots(repo)
    candidate_paths = candidate_roots_from(roots, limit)
    sizes, size_failures = measure_sizes(candidate_paths, mode=size_mode, timeout=size_timeout)
    open_pr_heads_cache: dict[str, list[dict[str, Any]]] | None = None
    open_pr_records_cache: list[dict[str, Any]] | None = None
    branch_pr_records_cache: dict[str, list[dict[str, Any]]] | None = None
    if include_pr_state:
        cache, records, branch_records, fetch_failed, _err = prefetch_open_pr_heads(
            repo, timeout=gh_timeout
        )
        if not fetch_failed:
            open_pr_heads_cache = cache
            open_pr_records_cache = records
            branch_pr_records_cache = branch_records
    context = InventoryContext(
        repo=repo,
        base=base,
        base_sha=base_sha,
        repo_remote_urls=repo_remote_urls(repo, timeout=git_timeout),
        strict_repo_identity=not explicit_roots,
        outbox_dir=outbox_dir if outbox_dir.is_absolute() else repo / outbox_dir,
        receipt_dir=receipt_dir if receipt_dir.is_absolute() else repo / receipt_dir,
        worktrees_by_path=parse_worktree_list(repo, timeout=git_timeout),
        unresolved_outbox_branches=unresolved_outbox_handoff_branches(
            repo,
            outbox_dir=outbox_dir,
            receipt_dir=receipt_dir,
        ),
        terminal_receipt_branch_heads=terminal_receipted_handoff_branch_heads(
            repo,
            outbox_dir=outbox_dir,
            receipt_dir=receipt_dir,
        ),
        terminal_receipt_path_heads=terminal_receipt_path_heads(
            [
                receipt_dir if receipt_dir.is_absolute() else repo / receipt_dir,
                repo / DEFAULT_HARVEST_RECEIPT_REL_DIR,
            ]
        ),
        skip_gh=skip_gh,
        git_timeout=git_timeout,
        gh_timeout=gh_timeout,
        patch_timeout=patch_timeout,
        smart_merge_detection=smart_merge_detection,
        smart_merge_main_subjects=(
            recent_main_commit_subjects(repo, base, timeout=git_timeout)
            if smart_merge_detection
            else []
        ),
        open_pr_heads_cache=open_pr_heads_cache,
        open_pr_records_cache=open_pr_records_cache,
        branch_pr_records_cache=branch_pr_records_cache,
    )
    candidates = [
        classify_candidate(
            path,
            context=context,
            size_bytes=sizes.get(str(path)),
            size_lookup_failed=str(path) in size_failures,
        )
        for path in candidate_paths
    ]
    now = utc_now().isoformat()
    payload = {
        "schema": SCHEMA,
        "generated_at": now,
        "root": str(roots[0]) if roots else "",
        "roots": [str(item) for item in roots],
        "repo": str(repo),
        "base": base,
        "base_sha": base_sha,
        "size_mode": size_mode,
        "smart_merge_detection": smart_merge_detection,
        "include_pr_state": include_pr_state,
        "open_pr_heads_cache_used": open_pr_heads_cache is not None,
        "limit": limit,
        "summary": build_summary(candidates),
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    return payload


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def write_ledger(ledger_root: Path, payload: dict[str, Any]) -> dict[str, str]:
    generated = str(payload["generated_at"]).replace(":", "").replace("-", "")
    snapshot_path = ledger_root / "snapshots" / f"{generated}.json"
    latest_path = ledger_root / "latest.json"
    ledger_path = ledger_root / "ledger.jsonl"
    snapshot_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write(snapshot_path, snapshot_text)
    atomic_write(latest_path, snapshot_text)
    event = {
        "schema": SCHEMA,
        "event_id": hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()[:16],
        "event_type": "inventory",
        "created_at": payload["generated_at"],
        "actor": "codex-worktree-value-inventory",
        "root": payload["root"],
        "summary": payload["summary"],
        "snapshot": str(snapshot_path),
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return {
        "snapshot": str(snapshot_path),
        "latest": str(latest_path),
        "ledger": str(ledger_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only inventory of Aragora worktree value and cleanup candidates. "
            "When --root is omitted, scans the canonical "
            "<repo>/.worktrees/codex-auto AND legacy ~/.codex/worktrees roots if "
            "either exists. Pass --root one or more times to override the default."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Inventory root directory. May be repeated to scan multiple roots. "
            "When omitted, defaults to the canonical Aragora worktree directory "
            "AND the legacy Codex Desktop directory if either exists."
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--outbox-dir", type=Path, default=DEFAULT_OUTBOX_DIR)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--ledger-root", type=Path, default=DEFAULT_LEDGER_ROOT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--size-mode", choices=("du", "stat", "none"), default="du")
    parser.add_argument("--size-timeout", type=int, default=300)
    parser.add_argument(
        "--git-timeout",
        "--git-timeout-seconds",
        dest="git_timeout",
        type=int,
        default=GIT_TIMEOUT_SECONDS,
        help=f"Timeout for each git subprocess (default {GIT_TIMEOUT_SECONDS}s; "
        "a timed-out candidate is annotated inspect_timeout and preserved)",
    )
    parser.add_argument("--gh-timeout", type=int, default=30)
    parser.add_argument("--patch-timeout", type=int, default=45)
    parser.add_argument("--skip-gh", action="store_true")
    parser.add_argument(
        "--smart-merge-detection",
        action="store_true",
        help=(
            "Reclassify ahead branches as patch_equivalent_or_merged when merge-tree "
            "or patch-presence proves the branch is already represented on base. "
            "Loose recent-main subject matches are recorded only as advisory context. "
            "Default off to preserve legacy inventory behavior."
        ),
    )
    parser.add_argument(
        "--include-pr-state",
        action="store_true",
        help=(
            "Supplement --skip-gh with a single cached `gh pr list --state open` "
            "lookup at scan start, mapping headRefName -> open PR records. "
            "Branches matching an open PR get classified as open_pr_or_outbox "
            "(preserved) rather than harvest_candidate. No-op if gh is unavailable."
        ),
    )
    parser.add_argument("--write-ledger", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Suppress ledger writes.")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.root:
        resolved_roots: list[Path] | None = [item.expanduser().resolve() for item in args.root]
    else:
        resolved_roots = None  # let inventory() call resolve_default_roots
    payload = inventory(
        roots=resolved_roots,
        repo=args.repo,
        base=args.base,
        outbox_dir=args.outbox_dir,
        receipt_dir=args.receipt_dir,
        limit=args.limit,
        size_mode=args.size_mode,
        size_timeout=args.size_timeout,
        skip_gh=args.skip_gh,
        git_timeout=args.git_timeout,
        gh_timeout=args.gh_timeout,
        patch_timeout=args.patch_timeout,
        smart_merge_detection=args.smart_merge_detection,
        include_pr_state=args.include_pr_state,
    )
    if args.write_ledger and not args.dry_run:
        payload["ledger_written"] = write_ledger(args.ledger_root, payload)
    else:
        payload["ledger_written"] = None

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload["summary"]
        roots_str = ", ".join(payload.get("roots") or []) or payload.get("root", "")
        print(f"roots: {roots_str}")
        print(f"candidates: {summary['total_candidates']}")
        print(f"coverage: {summary['inventory_coverage']:.2%}")
        print(f"cleanup_candidates: {summary['cleanup_candidate_count']}")
        print(f"harvest_candidates: {summary['harvest_candidate_count']}")
        print("classes:")
        for name, count in summary["count_by_class"].items():
            print(f"  {name}: {count}")
        print("safety_classes:")
        for name, count in summary["count_by_safety_class"].items():
            print(f"  {name}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
