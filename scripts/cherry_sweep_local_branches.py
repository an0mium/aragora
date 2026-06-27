#!/usr/bin/env python3
"""Patch-equivalence local-branch sweep using ``git cherry``.

This script implements the v12 P60 prompt-bug fix described in v13 P68:

    ``--merged`` and ``git rev-list --count`` miss squash-merged work
    because squashing rewrites the commit graph. The patch-equivalence
    detector built into ``git cherry`` survives squash-merge: it
    classifies each commit on ``<branch>`` as either ``-`` (a
    patch-equivalent commit already exists upstream) or ``+`` (no
    patch-equivalent on upstream). If *every* commit reachable from
    ``<branch>`` (relative to ``origin/main``) is patch-equivalent, the
    branch carries no unique work and is safe to delete (per R22).

Safety / rules from the v13 spec:

- **R19** — never amend pushed commits (not relevant here; we only
  invoke ``git branch -D``).
- **R20 / R25** — never use raw ``rm -rf`` on worktrees (not used here).
- **R22** — patch-equivalence via ``git cherry origin/main <branch>``.
- The script never touches ``main``, never touches any branch checked
  out by a worktree, never touches any branch claimed in the lane
  registry with an active-ish status.
- The script never deletes a branch whose configured upstream still
  exists on the remote (per the pre-flight rule in the task). A
  branch is considered "tracked remote still present" when
  ``%(upstream)`` is non-empty AND ``%(upstream:track)`` does **not**
  contain ``gone``.

Default mode is ``--dry-run`` (no mutation). ``--apply`` performs the
deletion using batched ``git branch -D`` calls. ``--limit N`` caps the
number of deletions in a single run.

Output is always a single JSON summary on stdout that conforms to:

    {
      "scanned": int,
      "skipped_main": int,
      "skipped_worktree": int,
      "skipped_claim": int,
      "skipped_tracked_remote": int,
      "skipped_error": int,
      "candidate_count": int,
      "deleted": int,
      "preserved_with_unique": int,
      "errors": [ {"branch": str, "phase": str, "stderr": str} ],
      "candidates": [str],          # only with --include-candidates
      "preserved": [str],           # only with --include-preserved
      "dry_run": bool,
      "applied": bool,
      "limit": int | null,
      "lane_registry": str | null,
      "active_lane_statuses": [str],
      "base": str,
      "deleted_branches": [str]     # populated on --apply
    }

The script is pure stdlib + ``git`` / ``gh``-free. It is designed to be
safe to invoke from any worktree of the repo, but **deletion is always
performed against the repo root that owns the configured ref store**
(i.e. it uses the same ``git`` binary that ``git for-each-ref`` returns
data from). It does not invoke any network operation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_LANE_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "lanes.json"
USER_LANE_PATH = Path.home() / ".aragora" / "agent-bridge" / "lanes.json"

# Statuses considered "claimed" — must mirror the constants in
# ``scripts/claim_active_agent_lane.py`` and ``scripts/sweep_stale_lane_claims.py``.
DEFAULT_ACTIVE_LANE_STATUSES: tuple[str, ...] = (
    "active",
    "running",
    "claimed",
    "pending",
    "queued",
)

# Branches that are *always* protected, even before any patch-equivalence
# check runs.
ALWAYS_PROTECT: frozenset[str] = frozenset({"main", "master"})

# Default batch size for ``git branch -D`` calls. Git's argv limit on
# macOS is ~256 KiB, so batches of 100 are well within bounds for
# branch names of any realistic length.
DEFAULT_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run_git(
    args: Sequence[str],
    *,
    repo: Path,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a git subprocess inside ``repo`` and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=check,
        capture_output=True,
        text=True,
    )


def list_local_branches(repo: Path) -> list[tuple[str, str, str]]:
    """Return ``[(name, upstream, track), ...]`` for every local ref.

    ``upstream`` is the full upstream refname (e.g.
    ``refs/remotes/origin/foo``) or the empty string when no upstream
    is configured. ``track`` is the parenthesised tracking summary from
    ``%(upstream:track)`` (``"[gone]"``, ``"[ahead 1]"``, ``""``, ...).
    """
    proc = _run_git(
        [
            "for-each-ref",
            "--format=%(refname:short)|%(upstream)|%(upstream:track)",
            "refs/heads/",
        ],
        repo=repo,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git for-each-ref failed: {proc.stderr.strip()}")
    rows: list[tuple[str, str, str]] = []
    for line in proc.stdout.splitlines():
        parts = line.split("|", 2)
        # Some refnames could theoretically contain ``|``; for-each-ref
        # in practice does not emit them in refs/heads, so the strict
        # 3-field split is enough.
        if not parts or not parts[0]:
            continue
        name = parts[0]
        upstream = parts[1] if len(parts) > 1 else ""
        track = parts[2] if len(parts) > 2 else ""
        rows.append((name, upstream, track))
    return rows


def worktree_bound_branches(repo: Path) -> set[str]:
    """Return the set of branch names that are HEADs of an existing worktree."""
    proc = _run_git(["worktree", "list", "--porcelain"], repo=repo)
    if proc.returncode != 0:
        raise RuntimeError(f"git worktree list failed: {proc.stderr.strip()}")
    branches: set[str] = set()
    for line in proc.stdout.splitlines():
        if line.startswith("branch "):
            ref = line[len("branch ") :].strip()
            if ref.startswith("refs/heads/"):
                ref = ref[len("refs/heads/") :]
            if ref:
                branches.add(ref)
    return branches


def cherry_classify(
    branch: str,
    *,
    base: str,
    repo: Path,
) -> tuple[bool, int, int, str]:
    """Run ``git cherry <base> <branch>`` and parse the output.

    Returns ``(all_patch_equivalent, unique_commit_count, equivalent_commit_count, stderr)``.
    ``all_patch_equivalent`` is True iff every output line begins with
    ``"-"`` (and there is at least one such line; an empty output also
    counts as patch-equivalent because the branch has no commits ahead
    of ``base``).
    """
    proc = _run_git(["cherry", base, branch], repo=repo)
    if proc.returncode != 0:
        return False, 0, 0, proc.stderr.strip()
    unique = 0
    equiv = 0
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        if line.startswith("+"):
            unique += 1
        elif line.startswith("-"):
            equiv += 1
        else:
            # Unrecognised marker — treat conservatively as unique.
            unique += 1
    all_equiv = unique == 0
    return all_equiv, unique, equiv, ""


# ---------------------------------------------------------------------------
# Lane registry helpers
# ---------------------------------------------------------------------------


def resolve_lane_registry(*, repo: Path, explicit: Path | None = None) -> Path | None:
    """Pick the canonical lane registry path, or ``None`` if missing.

    Prefers the explicit override > repo-local ``.aragora/agent-bridge/lanes.json``
    > user-level ``~/.aragora/agent-bridge/lanes.json``. Returns ``None`` when
    none of these exist (callers treat that as "no claims to respect").
    """
    if explicit is not None:
        return explicit if explicit.exists() else None
    candidate = repo / REPO_LANE_RELATIVE_PATH
    if candidate.exists():
        return candidate
    if USER_LANE_PATH.exists():
        return USER_LANE_PATH
    return None


def load_claimed_branches(
    registry_path: Path | None,
    *,
    active_statuses: Iterable[str] = DEFAULT_ACTIVE_LANE_STATUSES,
) -> set[str]:
    """Return the set of branch names currently claimed in the lane registry."""
    if registry_path is None:
        return set()
    try:
        data = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if not isinstance(data, list):
        return set()
    active_set = {s.lower() for s in active_statuses}
    claimed: set[str] = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).lower()
        if status not in active_set:
            continue
        branch = row.get("branch")
        if isinstance(branch, str) and branch:
            claimed.add(_normalize_branch(branch))
    return claimed


def _normalize_branch(name: str) -> str:
    if name.startswith("refs/heads/"):
        return name[len("refs/heads/") :]
    if name.startswith("origin/"):
        return name[len("origin/") :]
    return name


def has_tracked_remote_still_present(upstream: str, track: str) -> bool:
    """True when the branch's configured upstream exists on the remote.

    Per the lane spec: never delete a branch that has a tracked remote
    tracking ref unless the remote is also gone. ``%(upstream:track)``
    contains ``[gone]`` when the upstream remote ref has been deleted.
    """
    if not upstream:
        return False
    return "gone" not in track.lower()


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------


def sweep(
    *,
    repo: Path,
    base: str = "origin/main",
    apply: bool = False,
    limit: int | None = None,
    registry_path: Path | None = None,
    active_statuses: Iterable[str] = DEFAULT_ACTIVE_LANE_STATUSES,
    include_candidates: bool = False,
    include_preserved: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    branch_lister: Any = list_local_branches,
    worktree_lister: Any = worktree_bound_branches,
    cherry_runner: Any = cherry_classify,
    deleter: Any | None = None,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Run the patch-equivalence sweep.

    ``branch_lister``, ``worktree_lister``, ``cherry_runner`` and
    ``deleter`` are injection points for unit tests so the tests can
    fully exercise the sweep logic against a synthetic tmp repo without
    requiring ``git cherry`` semantics from a real bare repo.
    """
    active_statuses_tuple = tuple(sorted({s.lower() for s in active_statuses}))
    branches = branch_lister(repo)
    worktree_branches = worktree_lister(repo)
    claimed = load_claimed_branches(registry_path, active_statuses=active_statuses_tuple)

    skipped_main = 0
    skipped_worktree = 0
    skipped_claim = 0
    skipped_tracked_remote = 0
    skipped_error = 0
    candidates: list[str] = []
    preserved_with_unique = 0
    preserved: list[str] = []
    errors: list[dict[str, str]] = []

    for name, upstream, track in branches:
        if name in ALWAYS_PROTECT:
            skipped_main += 1
            continue
        if name in worktree_branches:
            skipped_worktree += 1
            continue
        if name in claimed:
            skipped_claim += 1
            continue
        if has_tracked_remote_still_present(upstream, track):
            skipped_tracked_remote += 1
            continue

        try:
            all_equiv, unique, _equiv, stderr = cherry_runner(name, base=base, repo=repo)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append({"branch": name, "phase": "cherry", "stderr": str(exc)})
            skipped_error += 1
            continue
        if stderr:
            errors.append({"branch": name, "phase": "cherry", "stderr": stderr})
            skipped_error += 1
            continue
        if all_equiv:
            candidates.append(name)
        else:
            preserved_with_unique += 1
            if include_preserved:
                preserved.append(name)

    deleted_branches: list[str] = []
    if apply and candidates:
        to_delete = candidates if limit is None else candidates[:limit]
        delete_fn = deleter if deleter is not None else _delete_branches
        deleted_branches = delete_fn(
            to_delete,
            repo=repo,
            batch_size=batch_size,
            errors=errors,
        )

    summary: dict[str, Any] = {
        "scanned": len(branches),
        "skipped_main": skipped_main,
        "skipped_worktree": skipped_worktree,
        "skipped_claim": skipped_claim,
        "skipped_tracked_remote": skipped_tracked_remote,
        "skipped_error": skipped_error,
        "candidate_count": len(candidates),
        "deleted": len(deleted_branches),
        "preserved_with_unique": preserved_with_unique,
        "errors": errors,
        "dry_run": not apply,
        "applied": apply,
        "limit": limit,
        "lane_registry": str(registry_path) if registry_path else None,
        "active_lane_statuses": list(active_statuses_tuple),
        "base": base,
        "deleted_branches": deleted_branches,
        "swept_at": (now or _utc_now()).isoformat().replace("+00:00", "Z"),
    }
    if include_candidates:
        summary["candidates"] = candidates
    if include_preserved:
        summary["preserved"] = preserved
    return summary


def _delete_branches(
    names: Sequence[str],
    *,
    repo: Path,
    batch_size: int,
    errors: list[dict[str, str]],
) -> list[str]:
    """Run ``git branch -D`` in batches; collect both deleted names and errors."""
    deleted: list[str] = []
    if not names:
        return deleted
    for chunk_start in range(0, len(names), batch_size):
        batch = list(names[chunk_start : chunk_start + batch_size])
        proc = _run_git(["branch", "-D", *batch], repo=repo)
        # ``git branch -D`` succeeds only when every branch was deleted.
        # On partial failure it emits per-line success/failure to stdout
        # and stderr, so we parse both streams.
        succeeded = set()
        for line in proc.stdout.splitlines():
            stripped = line.strip()
            # Lines look like: ``Deleted branch foo (was abc1234).``
            if stripped.startswith("Deleted branch "):
                rest = stripped[len("Deleted branch ") :]
                # Drop the " (was ...)." suffix.
                paren = rest.find(" (was ")
                if paren > 0:
                    succeeded.add(rest[:paren])
                else:
                    succeeded.add(rest.split()[0])
        for line in proc.stderr.splitlines():
            stripped = line.strip()
            if stripped:
                errors.append({"branch": "", "phase": "delete", "stderr": stripped})
        for b in batch:
            if b in succeeded:
                deleted.append(b)
    return deleted


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Patch-equivalence local-branch sweep using `git cherry`. Default is dry-run.")
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run `git branch -D` for every candidate (default: dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicit dry-run; mutually exclusive with --apply (default behaviour).",
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Patch-equivalence base ref (default: origin/main).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of branches deleted in this run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"`git branch -D` batch size (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root to operate on (default: this script's repo).",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Explicit override for the lane registry JSON path.",
    )
    parser.add_argument(
        "--include-candidates",
        action="store_true",
        help="Include the full candidate list in the JSON summary.",
    )
    parser.add_argument(
        "--include-preserved",
        action="store_true",
        help="Include preserved-branch names in the JSON summary.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, write the JSON summary to this path in addition to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.apply and args.dry_run:
        print("ERROR: --apply and --dry-run are mutually exclusive", file=sys.stderr)
        return 2

    repo = args.repo_root.resolve()
    registry_path = resolve_lane_registry(repo=repo, explicit=args.registry_path)

    summary = sweep(
        repo=repo,
        base=args.base,
        apply=args.apply,
        limit=args.limit,
        registry_path=registry_path,
        include_candidates=args.include_candidates,
        include_preserved=args.include_preserved,
        batch_size=args.batch_size,
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
