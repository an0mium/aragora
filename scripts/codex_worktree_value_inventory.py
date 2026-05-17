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
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_codex_branch_backlog import (  # noqa: E402
    DEFAULT_OUTBOX_DIR,
    DEFAULT_RECEIPT_DIR,
    _commit_prefix_matches,
    is_patch_equivalent,
    terminal_receipted_handoff_branch_heads,
    unresolved_outbox_handoff_branches,
)

SCHEMA = "aragora-worktree-harvest/1.0"
DEFAULT_LEGACY_ROOT = Path.home() / ".codex" / "worktrees"
DEFAULT_CANONICAL_REL_ROOT = Path(".worktrees") / "codex-auto"
DEFAULT_ROOT = DEFAULT_LEGACY_ROOT  # kept for backward compatibility
DEFAULT_LEDGER_ROOT = Path(".aragora/worktree-harvest")
ACTIVE_SESSION_FILES = (
    ".claude-session-active",
    ".codex_session_active",
    ".nomic-session-active",
)
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
    lookup_failed: bool = False
    lookup_errors: list[str] = field(default_factory=list)


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


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def run_cmd(args: list[str], cwd: Path, *, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        stderr = str(exc)
        if isinstance(exc, subprocess.TimeoutExpired):
            stderr = f"command timed out after {timeout}s: {' '.join(args)}"
        return subprocess.CompletedProcess(args=args, returncode=124, stdout="", stderr=stderr)


def run_git(args: list[str], cwd: Path, *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
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
    if active_lock_files(candidate_root, repo_path):
        return True
    for root in (candidate_root, repo_path):
        if root is None or not root.exists():
            continue
        try:
            for item in root.glob(".claude-session-anchor"):
                if item.exists():
                    return True
        except OSError:
            return True
    return False


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


def lookup_open_prs(
    repo: Path, branch: str | None, *, timeout: int, skip_gh: bool
) -> tuple[list[dict[str, Any]], bool, str | None]:
    if not branch or skip_gh:
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
    )
    links["open_prs"] = open_prs
    if open_pr_failed:
        git.lookup_failed = True
        git.lookup_errors.append(open_pr_error or "open PR lookup failed")

    links["outbox_files"] = outbox_files_for_branch(context.outbox_dir, branch)
    links["receipt_files"] = receipt_files_for_branch(context.receipt_dir, branch)
    outbox_protected = bool(branch and branch in context.unresolved_outbox_branches)
    receipt_protected = branch_matches_receipt(
        branch,
        head,
        context.terminal_receipt_branch_heads,
    )

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
        proof.append("terminal automation receipt references branch/head")
    elif git.ahead and git.ahead > 0:
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
            else:
                classification = "unique_unharvested"
                proof.append("branch has unique commits or diff ahead of base")
    elif git.registered_worktree:
        classification = "patch_equivalent_or_merged"
        proof.append("registered git worktree has no unique commits ahead of base")
    else:
        classification = "unregistered_git_residue"
        proof.append("git checkout is not registered in git worktree list")

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
        proof=proof,
        active_session=active_session,
        lock_files=lock_files,
        git=git,
        links=links,
        next_action=next_action,
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
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
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
                "size_bytes": candidate.size_bytes,
                "branch": candidate.git.branch,
                "head": candidate.git.head,
                "decision": candidate.decision,
                "proof": candidate.proof,
            }
            for candidate in selected[:20]
        ]

    return {
        "total_candidates": len(candidates),
        "classified_candidates": sum(counts.values()),
        "unknown_candidates": counts.get("lookup_failed", 0),
        "count_by_class": {name: counts.get(name, 0) for name in VALUE_CLASSES},
        "bytes_by_class": bytes_by_class,
        "known_size_bytes": known_bytes,
        "size_lookup_failures": size_lookup_failures,
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
        skip_gh=skip_gh,
        git_timeout=git_timeout,
        gh_timeout=gh_timeout,
        patch_timeout=patch_timeout,
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
    parser.add_argument("--git-timeout", type=int, default=30)
    parser.add_argument("--gh-timeout", type=int, default=30)
    parser.add_argument("--patch-timeout", type=int, default=45)
    parser.add_argument("--skip-gh", action="store_true")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
