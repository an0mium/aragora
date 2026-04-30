#!/usr/bin/env python3
"""Guarded worktree inspection and cleanup for ad-hoc side branches."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import codex_worktree_autopilot as autopilot


@dataclass
class WorktreeInspection:
    path: str
    exists: bool
    tracked_worktree: bool
    branch: str | None
    active_session: bool
    lock_files: list[str]
    dirty: bool
    unique_commits_ahead: int
    ahead_lookup_failed: bool
    open_prs: list[dict[str, Any]]
    pr_lookup_failed: bool
    blockers: list[str]


def _active_lock_files(path: Path) -> list[str]:
    return [
        name
        for name in (".claude-session-active", ".codex_session_active", ".nomic-session-active")
        if (path / name).exists()
    ]


def _get_worktree_entry(repo_root: Path, path: Path) -> autopilot.WorktreeEntry | None:
    target = path.resolve()
    for entry in autopilot._get_worktree_entries(repo_root):
        if entry.path.resolve() == target:
            return entry
    return None


def _branch_for_path(path: Path, entry: autopilot.WorktreeEntry | None) -> str | None:
    if entry and entry.branch:
        return entry.branch
    if not path.exists():
        return None
    if not (path / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    branch = proc.stdout.strip()
    return None if not branch or branch == "HEAD" else branch


def _lookup_open_prs(repo_root: Path, branch: str | None) -> tuple[list[dict[str, Any]], bool]:
    if not branch:
        return [], False
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--head",
                branch,
                "--json",
                "number,title,url",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return [], True
    if proc.returncode != 0:
        return [], True
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError:
        return [], True
    if not isinstance(payload, list):
        return [], True
    return payload, False


def _worktree_is_dirty(path: Path) -> bool:
    if not path.exists():
        return False
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())


def _unique_commits_ahead_of_main(
    repo_root: Path,
    branch: str | None,
) -> tuple[int, bool]:
    if not branch:
        return 0, False
    proc = autopilot._run_git(repo_root, "rev-list", "--count", f"origin/main..{branch}")
    if proc.returncode != 0:
        return 0, True
    try:
        return int(proc.stdout.strip() or "0"), False
    except ValueError:
        return 0, True


def _pr_lookup_failure_blocks(
    branch: str | None,
    *,
    unique_commits_ahead: int,
    ahead_lookup_failed: bool,
) -> bool:
    if not branch:
        return False
    if ahead_lookup_failed:
        return True
    return unique_commits_ahead > 0


def inspect_worktree(
    repo_root: Path, path: Path, *, branch_override: str | None = None
) -> WorktreeInspection:
    path = path.resolve()
    exists = path.exists()
    entry = _get_worktree_entry(repo_root, path)
    tracked_worktree = entry is not None
    branch = branch_override or _branch_for_path(path, entry)
    active_session = exists and autopilot._has_active_session(path)
    lock_files = _active_lock_files(path) if exists else []
    dirty = _worktree_is_dirty(path) if exists else False
    unique_commits_ahead, ahead_lookup_failed = _unique_commits_ahead_of_main(repo_root, branch)
    open_prs, pr_lookup_failed = _lookup_open_prs(repo_root, branch)

    blockers: list[str] = []
    if not exists:
        blockers.append("missing_path")
    if active_session:
        blockers.append("active_session")
    if lock_files:
        blockers.append("session_lock_present")
    if dirty:
        blockers.append("dirty_worktree")
    if unique_commits_ahead > 0:
        blockers.append("branch_ahead_of_origin_main")
    if open_prs:
        blockers.append("open_pr")
    if branch and ahead_lookup_failed:
        blockers.append("ahead_lookup_failed")
    if pr_lookup_failed and _pr_lookup_failure_blocks(
        branch,
        unique_commits_ahead=unique_commits_ahead,
        ahead_lookup_failed=ahead_lookup_failed,
    ):
        blockers.append("pr_lookup_failed")

    return WorktreeInspection(
        path=str(path),
        exists=exists,
        tracked_worktree=tracked_worktree,
        branch=branch,
        active_session=active_session,
        lock_files=lock_files,
        dirty=dirty,
        unique_commits_ahead=unique_commits_ahead,
        ahead_lookup_failed=ahead_lookup_failed,
        open_prs=open_prs,
        pr_lookup_failed=pr_lookup_failed,
        blockers=blockers,
    )


def _print_inspection(inspection: WorktreeInspection, *, as_json: bool) -> None:
    payload = asdict(inspection)
    payload["removable"] = not inspection.blockers
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    print(f"path: {inspection.path}")
    print(f"exists: {inspection.exists}")
    print(f"tracked_worktree: {inspection.tracked_worktree}")
    print(f"branch: {inspection.branch or '-'}")
    print(f"active_session: {inspection.active_session}")
    if inspection.lock_files:
        print(f"lock_files: {', '.join(inspection.lock_files)}")
    print(f"dirty: {inspection.dirty}")
    if inspection.branch:
        print(f"unique_commits_ahead: {inspection.unique_commits_ahead}")
        print(f"ahead_lookup_failed: {inspection.ahead_lookup_failed}")
    print(f"open_prs: {len(inspection.open_prs)}")
    if inspection.open_prs:
        for pr in inspection.open_prs:
            print(f"  - #{pr.get('number')} {pr.get('title')} :: {pr.get('url')}")
    print(f"removable: {not inspection.blockers}")
    if inspection.blockers:
        print("blockers:")
        for blocker in inspection.blockers:
            print(f"  - {blocker}")


def _delete_branch(repo_root: Path, branch: str) -> bool:
    if not autopilot._branch_exists(repo_root, branch):
        return True
    proc = autopilot._run_git(repo_root, "branch", "-D", branch)
    return proc.returncode == 0


def remove_worktree(
    repo_root: Path,
    inspection: WorktreeInspection,
    *,
    delete_branch: bool,
    purge_path: bool,
    force: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": inspection.path,
        "branch": inspection.branch,
        "removed": False,
        "branch_deleted": False,
        "path_purged": False,
        "blockers": list(inspection.blockers),
    }
    path = Path(inspection.path)
    if inspection.blockers and not force:
        result["status"] = "blocked"
        return result

    if inspection.tracked_worktree:
        proc = autopilot._run_git(repo_root, "worktree", "remove", "--force", inspection.path)
        if proc.returncode != 0:
            result["status"] = "remove_failed"
            result["stderr"] = proc.stderr.strip()
            if not purge_path:
                return result
        else:
            result["removed"] = True
            result["status"] = "removed"
    else:
        result["status"] = "untracked_path"
        if not purge_path:
            return result

    if path.exists() and purge_path:
        shutil.rmtree(path, ignore_errors=True)
        result["path_purged"] = not path.exists()
        if result["path_purged"]:
            result["removed"] = True

    if delete_branch and inspection.branch:
        result["branch_deleted"] = _delete_branch(repo_root, inspection.branch)

    if result.get("status") in {"removed", "untracked_path"} and not result["removed"]:
        result["status"] = "partial"
    elif result.get("status") == "untracked_path" and result["removed"]:
        result["status"] = "purged"
    elif result.get("status") == "remove_failed" and result["path_purged"]:
        result["status"] = "purged_after_failed_remove"

    return result


def _repo_root_from_arg(repo: str) -> Path:
    return autopilot._repo_root_from(Path(repo))


def cmd_inspect(args: argparse.Namespace) -> int:
    repo_root = _repo_root_from_arg(args.repo)
    inspection = inspect_worktree(repo_root, Path(args.path), branch_override=args.branch)
    _print_inspection(inspection, as_json=args.json)
    return 0 if not inspection.blockers else 1


def cmd_remove(args: argparse.Namespace) -> int:
    repo_root = _repo_root_from_arg(args.repo)
    inspection = inspect_worktree(repo_root, Path(args.path), branch_override=args.branch)
    result = remove_worktree(
        repo_root,
        inspection,
        delete_branch=args.delete_branch,
        purge_path=args.purge_path,
        force=args.force,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))
    status = str(result.get("status", ""))
    if status in {"blocked", "remove_failed", "untracked_path", "partial"}:
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and safely remove ad-hoc worktrees.")
    parser.add_argument("--repo", default=".", help="Path inside the target repository")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect a worktree for active-session / open-PR blockers"
    )
    inspect_parser.add_argument("path", help="Worktree path to inspect")
    inspect_parser.add_argument(
        "--branch", help="Override the branch name for orphaned or partially deleted paths"
    )
    inspect_parser.add_argument("--json", action="store_true")
    inspect_parser.set_defaults(func=cmd_inspect)

    remove_parser = subparsers.add_parser(
        "remove", help="Safely remove a worktree if no blockers exist"
    )
    remove_parser.add_argument("path", help="Worktree path to remove")
    remove_parser.add_argument(
        "--branch", help="Override the branch name for orphaned or partially deleted paths"
    )
    remove_parser.add_argument(
        "--delete-branch",
        action="store_true",
        help="Delete the local branch after removing the worktree",
    )
    remove_parser.add_argument(
        "--purge-path",
        action="store_true",
        help="Delete a residual path if git worktree removal leaves files behind",
    )
    remove_parser.add_argument(
        "--force", action="store_true", help="Bypass active-session/open-PR blockers"
    )
    remove_parser.add_argument("--json", action="store_true")
    remove_parser.set_defaults(func=cmd_remove)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
