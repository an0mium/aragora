#!/usr/bin/env python3
"""Parallel branch backlog classifier (Phase 2 of cleanup plan).

Wraps scripts/audit_codex_branch_backlog.py helpers but parallelises the
slow per-branch git operations (git cherry, git diff | git patch-id) so
the full 1,600+ branch audit runs in ~5 minutes instead of ~30 minutes.

Output: JSON file at .aragora/cleanup-state/branch-classification-<ts>.json
with one entry per branch matching the audit script's category vocabulary.

Re-running with the same output path resumes from cached results — only
branches whose head_sha changed since the last run get reclassified.

Usage:
    python3 scripts/audit_branch_backlog_parallel.py \\
        --repo /Users/armand/Development/aragora \\
        --base origin/main \\
        --prefix codex/ \\
        [--workers 8] \\
        [--out .aragora/cleanup-state/branch-classification.json] \\
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import helpers from the existing audit script (unchanged).
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_codex_branch_backlog import (  # noqa: E402
    ACTIVE_SESSION_FILES,
    branch_patch_id,
    count_ahead,
    is_patch_equivalent,
    local_branches,
    merged_branch_names,
    open_pr_heads,
    remote_branch_names,
    run_git,
    terminal_handoff_keys,
    terminal_receipted_handoff_branches,
    unresolved_outbox_handoff_branches,
    worktree_map,
)
from github_cli_health import check_github_cli_health  # noqa: E402

UTC = timezone.utc

CATEGORY_PROTECTED_OPEN_PR = "protected_open_pr"
CATEGORY_PROTECTED_ACTIVE_WT = "protected_active_worktree"
CATEGORY_PROTECTED_DIRTY_WT = "protected_dirty_worktree"
CATEGORY_PROTECTED_HANDOFF_RECEIPT = "protected_handoff_receipt"
CATEGORY_PROTECTED_HANDOFF_OUTBOX = "protected_handoff_outbox"
CATEGORY_CLEANUP_LOCAL_MERGED = "cleanup_local_merged"
CATEGORY_CLEANUP_PATCH_EQUIVALENT = "cleanup_patch_equivalent"
CATEGORY_SALVAGE_RECENT_UNIQUE = "salvage_recent_unique"
CATEGORY_SALVAGE_STALE_REMOTE_UNIQUE = "salvage_stale_remote_unique"
CATEGORY_SALVAGE_STALE_LOCAL_UNIQUE = "salvage_stale_local_unique"
EXTRA_ACTIVE_SESSION_MARKERS = (
    ".codex-session-active",
    ".droid-session-active",
    ".aragora-session.lock",
)
ACTIVE_SESSION_MARKERS = tuple(
    dict.fromkeys((*ACTIVE_SESSION_FILES, *EXTRA_ACTIVE_SESSION_MARKERS))
)


def _parse_iso(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


def _is_dirty(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        proc = run_git(["status", "--porcelain"], path, timeout=15)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    if proc.returncode != 0:
        return True
    return bool(proc.stdout.strip())


def _has_active_session(path: Path) -> bool:
    """Lightweight active-session check (lock files + session marker files)."""
    if not path.exists():
        return False
    for marker in ACTIVE_SESSION_MARKERS:
        if (path / marker).exists():
            return True
    return False


def _patch_equiv_worker(args: tuple[Path, str, str]) -> tuple[str, bool]:
    root, base, branch = args
    return branch, is_patch_equivalent(root, base, branch)


def _patch_id_worker(args: tuple[Path, str, str]) -> tuple[str, str | None]:
    root, base, branch = args
    return branch, branch_patch_id(root, base, branch)


def _classify_one(
    *,
    branch_row: dict[str, Any],
    open_pr_branches: dict[str, int],
    receipted_branches: set[str],
    outbox_branches: set[str],
    worktrees: dict[str, list[Path]],
    merged: set[str],
    remotes: set[str],
    patch_equivalents: dict[str, bool],
    handoff_patch_ids: set[str],
    branch_patch_ids_map: dict[str, str | None],
    recent_threshold: datetime,
) -> dict[str, Any]:
    branch = branch_row["name"]
    head_sha = branch_row["head_sha"]
    committed_at = _parse_iso(branch_row["committed_at"])
    ahead_count = int(branch_row.get("ahead_count") or 0)
    wts = worktrees.get(branch, [])

    pr_number = open_pr_branches.get(branch)
    is_dirty = any(_is_dirty(wt) for wt in wts)
    is_active = any(_has_active_session(wt) for wt in wts)
    in_receipts = branch in receipted_branches
    in_outbox = branch in outbox_branches
    pid = branch_patch_ids_map.get(branch)
    has_handoff_patch_match = pid is not None and pid in handoff_patch_ids
    is_pe = patch_equivalents.get(branch, False)
    in_merged = branch in merged
    on_remote = branch in remotes

    if pr_number is not None:
        category = CATEGORY_PROTECTED_OPEN_PR
    elif is_active:
        category = CATEGORY_PROTECTED_ACTIVE_WT
    elif is_dirty:
        category = CATEGORY_PROTECTED_DIRTY_WT
    elif in_receipts or has_handoff_patch_match:
        category = CATEGORY_PROTECTED_HANDOFF_RECEIPT
    elif in_outbox:
        category = CATEGORY_PROTECTED_HANDOFF_OUTBOX
    elif in_merged:
        category = CATEGORY_CLEANUP_LOCAL_MERGED
    elif is_pe:
        category = CATEGORY_CLEANUP_PATCH_EQUIVALENT
    elif ahead_count > 0:
        if committed_at >= recent_threshold:
            category = CATEGORY_SALVAGE_RECENT_UNIQUE
        elif on_remote:
            category = CATEGORY_SALVAGE_STALE_REMOTE_UNIQUE
        else:
            category = CATEGORY_SALVAGE_STALE_LOCAL_UNIQUE
    else:
        # No unique commits, no PR, not protected, not merged-or-equivalent.
        # Treat as patch-equivalent for cleanup purposes.
        category = CATEGORY_CLEANUP_PATCH_EQUIVALENT

    return {
        "branch": branch,
        "head_sha": head_sha,
        "committed_at": branch_row["committed_at"],
        "category": category,
        "pr_number": pr_number,
        "ahead_count": ahead_count,
        "patch_equivalent": is_pe,
        "merged_to_base": in_merged,
        "on_remote": on_remote,
        "worktree_paths": [str(p) for p in wts],
        "dirty": is_dirty,
        "active_session": is_active,
        "in_receipt_set": in_receipts,
        "in_outbox_set": in_outbox,
        "patch_id": pid,
        "subject": branch_row.get("subject", ""),
    }


def _load_resume_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    records = data.get("records") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return {}
    return {r["branch"]: r for r in records if isinstance(r, dict) and "branch" in r}


def _load_open_prs(
    root: Path,
    repo_name: str,
    prefix: str,
) -> tuple[dict[str, int], dict[str, Any], bool]:
    """Fetch open PR heads only when the GitHub health probe is ready."""

    try:
        health = check_github_cli_health(root)
    except Exception as exc:
        return (
            {},
            {
                "ready": False,
                "auth_ok": False,
                "api_ok": False,
                "mode": "health_check_failed",
                "error": str(exc),
                "repo": str(root),
            },
            True,
        )

    health_payload = health.to_dict()
    if not health.ready:
        return {}, health_payload, True
    return open_pr_heads(root, repo_name, prefix), health_payload, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help=(
            "Main repository root (NOT a worktree). The script reads "
            "automation outbox/receipt state from this path's .aragora/ "
            "directory; pointing it at a worktree may use stale or empty "
            "state and miss live handoff protection."
        ),
    )
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--prefix", default="codex/")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out",
        default=".aragora/cleanup-state/branch-classification.json",
        help="Output classification JSON path",
    )
    parser.add_argument(
        "--repo-name", default="synaptent/aragora", help="GitHub repo for PR lookup"
    )
    parser.add_argument("--recent-hours", type=int, default=72, help="Cutoff for recent salvage")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse cached classifications when head_sha unchanged",
    )
    parser.add_argument(
        "--max-branches", type=int, default=None, help="Cap branches scanned (testing)"
    )
    args = parser.parse_args(argv)

    root = Path(args.repo).resolve()
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts_start = datetime.now(UTC)
    print(f"[{ts_start.isoformat()}] Enumerating branches with prefix {args.prefix!r}")

    rows = local_branches(root, args.prefix, args.base)
    if args.max_branches:
        rows = rows[: args.max_branches]
    print(f"  found {len(rows)} branches")

    print("  enumerating remotes + merged + worktrees")
    remotes = remote_branch_names(root, args.prefix)
    merged = merged_branch_names(root, args.base, args.prefix)
    worktrees = worktree_map(root)
    print(
        f"  remotes={len(remotes)} merged={len(merged)} worktree-mapped-branches={len(worktrees)}"
    )

    print("  checking GitHub health for open PR lookup")
    open_prs, github_health, open_pr_lookup_skipped = _load_open_prs(
        root, args.repo_name, args.prefix
    )
    if open_pr_lookup_skipped:
        mode = github_health.get("mode", "unknown")
        error = " ".join(str(github_health.get("error", "")).split())
        print(f"  open PR lookup skipped: GitHub unavailable [{mode}] {error}".rstrip())
    else:
        print(f"  open_prs={len(open_prs)}")

    print("  loading automation outbox + receipt protection")
    receipted = terminal_receipted_handoff_branches(root)
    outbox = unresolved_outbox_handoff_branches(root)
    handoff_keys = terminal_handoff_keys(root / ".aragora" / "automation-receipts")
    print(
        f"  receipted_branches={len(receipted)} unresolved_outbox_branches={len(outbox)} "
        f"terminal_handoff_keys={len(handoff_keys)}"
    )

    cache = _load_resume_cache(out_path) if args.resume else {}
    if cache:
        print(f"  resume cache loaded: {len(cache)} prior records")

    branches_needing_check = [
        r
        for r in rows
        if not (args.resume and cache.get(r["name"], {}).get("head_sha") == r["head_sha"])
    ]
    print(
        f"  branches needing fresh patch-equivalence check: {len(branches_needing_check)} "
        f"(rest reused from cache)"
    )

    work_args = [(root, args.base, r["name"]) for r in branches_needing_check]

    patch_equivalents: dict[str, bool] = {
        b: cache[b]["patch_equivalent"] for b in cache if "patch_equivalent" in cache[b]
    }
    branch_pids: dict[str, str | None] = {b: cache[b].get("patch_id") for b in cache}

    if work_args:
        print(
            f"  parallelising {len(work_args)} patch-equivalence checks with {args.workers} workers"
        )
        completed_eq = 0
        with ThreadPoolExecutor(max_workers=args.workers) as eq_pool:
            patch_equiv_futures = [eq_pool.submit(_patch_equiv_worker, wa) for wa in work_args]
            for pe_future in as_completed(patch_equiv_futures):
                branch, is_pe = pe_future.result()
                patch_equivalents[branch] = is_pe
                completed_eq += 1
                if completed_eq % 100 == 0:
                    print(f"    {completed_eq}/{len(work_args)} patch-equiv checks done")

        print(
            f"  parallelising {len(work_args)} patch-id computations (needed for handoff matching)"
        )
        completed_pid = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pid_pool:
            patch_id_futures = [pid_pool.submit(_patch_id_worker, wa) for wa in work_args]
            for pid_future in as_completed(patch_id_futures):
                branch, pid = pid_future.result()
                branch_pids[branch] = pid
                completed_pid += 1
                if completed_pid % 100 == 0:
                    print(f"    {completed_pid}/{len(work_args)} patch-id computations done")

    handoff_patch_ids = {
        pid
        for branch, pid in branch_pids.items()
        if branch in receipted or branch in outbox
        if pid is not None
    }

    from datetime import timedelta

    recent_threshold = datetime.now(UTC) - timedelta(hours=args.recent_hours)

    print(f"  classifying {len(rows)} branches")
    records = []
    for row in rows:
        rec = _classify_one(
            branch_row=row,
            open_pr_branches=open_prs,
            receipted_branches=receipted,
            outbox_branches=outbox,
            worktrees=worktrees,
            merged=merged,
            remotes=remotes,
            patch_equivalents=patch_equivalents,
            handoff_patch_ids=handoff_patch_ids,
            branch_patch_ids_map=branch_pids,
            recent_threshold=recent_threshold,
        )
        records.append(rec)

    summary: dict[str, int] = {}
    for r in records:
        summary[r["category"]] = summary.get(r["category"], 0) + 1

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "started_at": ts_start.isoformat(),
        "repo": str(root),
        "base": args.base,
        "prefix": args.prefix,
        "github_health": github_health,
        "open_pr_lookup_skipped": open_pr_lookup_skipped,
        "totals": {
            "branches_scanned": len(rows),
            "branches_with_open_pr": len(open_prs),
            "remote_branches": len(remotes),
            "merged_branches": len(merged),
            "receipted_handoff_branches": len(receipted),
            "unresolved_outbox_branches": len(outbox),
        },
        "summary_by_category": summary,
        "records": records,
    }

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    elapsed = (datetime.now(UTC) - ts_start).total_seconds()
    print(f"\n[{datetime.now(UTC).isoformat()}] DONE in {elapsed:.1f}s")
    print(f"  output: {out_path}")
    print(f"  totals: {payload['totals']}")
    print("  by category:")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]):
        print(f"    {v:>5}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
